"""
Step 3: pretrain the MoE model on the mixed Japanese/English binary. (macOS version of ../train.py)

  python macosx/train.py --preset small --max_steps 20000     # run from the repository root

Differences from ../train.py:
  - uses the Apple GPU (MPS) when available, otherwise the CPU (with --dtype forced to fp32)
  - GradScaler (--dtype fp16) is created for the selected device
  - imports macosx/moe_model.py (real-valued RoPE)

Output during training:
  - every log_interval : loss / load-balancing loss / learning rate / tokens/s / expert utilization (all-layer average)
  - every eval_interval: validation loss (mixed, Japanese only, English only) and per-language, per-layer expert utilization
  - every sample_interval: text generation from prompts
Logs are also appended to out/train_log.jsonl.

Checkpoints:
  out/ckpt.pt      ... latest (saved at each eval / on Ctrl+C). Continue with --resume
  out/ckpt_best.pt ... the one with the best validation loss
"""
import argparse
import json
import math
import os
import time

import numpy as np
import sentencepiece as spm
import torch

from moe_model import PRESETS, MoEConfig, MoETransformer, load_checkpoint


class BinLoader:
    """Build mini-batches by slicing random positions from a raw binary (uint16 / uint32)."""

    def __init__(self, path, block_size, batch_size, device, dtype=np.uint16):
        self.path, self.block_size, self.batch_size, self.device, self.dtype = path, block_size, batch_size, device, dtype
        self.n_tokens = os.path.getsize(path) // np.dtype(dtype).itemsize
        assert self.n_tokens > block_size + 1, f"{path} has too few tokens"

    def get_batch(self):
        # recreate the memmap every time (nanoGPT's trick to avoid memory growth in long runs)
        data = np.memmap(self.path, dtype=self.dtype, mode="r")
        ix = np.random.randint(0, self.n_tokens - self.block_size - 1, size=self.batch_size)
        x = torch.from_numpy(np.stack([data[i:i + self.block_size] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + self.block_size] for i in ix]).astype(np.int64))
        if self.device.type == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


def get_lr(step, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    ratio = min(1.0, (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps))
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (args.lr - args.min_lr)


def fmt_util(counts):
    pct = counts / counts.sum() * 100
    return " ".join(f"{p:4.1f}" for p in pct.tolist())


@torch.no_grad()
def evaluate(model, loaders, args, ctx):
    model.eval()
    result = {}
    for name, loader in loaders.items():
        losses, counts = [], 0
        for _ in range(args.eval_iters):
            x, y = loader.get_batch()
            with ctx:
                out = model(x, y)
            losses.append(out["ce"].item())
            counts = counts + out["counts"].cpu()  # [n_layer, n_expert]
        result[name] = {"loss": float(np.mean(losses)), "counts": counts}
    model.train()
    return result


@torch.no_grad()
def sample(model, sp, prompts, device, ctx, max_new_tokens):
    model.eval()
    for p in prompts:
        idx = torch.tensor([[sp.bos_id()] + sp.encode(p)], device=device)
        with ctx:
            out = model.generate(idx, max_new_tokens, temperature=0.8, top_k=50, eos_id=sp.eos_id())
        print(f"  > {sp.decode(out[0].tolist()[1:])!r}")
    model.train()


def save_ckpt(path, model, optimizer, step, best_val, args):
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model.cfg.to_dict(),
        "step": step,
        "best_val": best_val,
        "args": vars(args),
    }, tmp)
    os.replace(tmp, path)  # an interrupted write won't corrupt the existing file


def main():
    ap = argparse.ArgumentParser(description="Train an MoE model on Japanese/English Wikipedia")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--preset", default="small", choices=list(PRESETS))
    # per-field model shape overrides (preset values if not given)
    for k in ("block_size", "n_layer", "n_head", "n_kv_head", "d_model", "d_ff", "n_expert", "n_expert_used"):
        ap.add_argument(f"--{k}", type=int, default=None)
    ap.add_argument("--aux_loss_coef", type=float, default=0.01)
    ap.add_argument("--tie_embeddings", action="store_true")
    # training settings
    ap.add_argument("--batch_size", type=int, default=8, help="micro-batch size (small + 32k vocabulary uses ~10GB VRAM)")
    ap.add_argument("--grad_accum", type=int, default=8, help="gradient accumulation steps (effective batch = batch_size * grad_accum)")
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min_lr", type=float, default=6e-5)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--compile", action="store_true", help="use torch.compile (requires Triton on Windows)")
    ap.add_argument("--seed", type=int, default=1337)
    # monitoring
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_iters", type=int, default=20)
    ap.add_argument("--sample_interval", type=int, default=1000)
    ap.add_argument("--sample_tokens", type=int, default=48)
    ap.add_argument("--prompts", nargs="+", default=["日本の首都は", "The history of"])
    ap.add_argument("--resume", action="store_true", help="resume from out_dir/ckpt.pt")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "cpu" and args.dtype != "fp32":
        args.dtype = "fp32"  # bf16/fp16 autocast on the CPU is much slower than fp32
    print(f"Device: {device} / dtype: {args.dtype}")

    with open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    sp = spm.SentencePieceProcessor(model_file=os.path.join(args.data_dir, "tokenizer.model"))

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    start_step, best_val = 0, float("inf")
    if args.resume:
        model, ckpt = load_checkpoint(ckpt_path, device)
        start_step, best_val = ckpt["step"], ckpt.get("best_val", float("inf"))
        print(f"Resuming: {ckpt_path} (step {start_step})")
    else:
        cfg_dict = dict(PRESETS[args.preset], vocab_size=meta["vocab_size"],
                        aux_loss_coef=args.aux_loss_coef, tie_embeddings=args.tie_embeddings)
        for k in ("block_size", "n_layer", "n_head", "n_kv_head", "d_model", "d_ff", "n_expert", "n_expert_used"):
            if getattr(args, k) is not None:
                cfg_dict[k] = getattr(args, k)
        model = MoETransformer(MoEConfig(**cfg_dict)).to(device)
    cfg = model.cfg
    total, active = model.num_params()
    print(f"Model: {cfg}")
    print(f"Parameters: total {total / 1e6:.1f}M / active per token {active / 1e6:.1f}M")

    # weight decay only for tensors with 2+ dims (matrices, expert tensors, embeddings); not RMSNorm
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=device.type == "cuda",
    )
    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer"])
        del ckpt

    pt_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    ctx = torch.autocast(device_type=device.type, dtype=pt_dtype, enabled=args.dtype != "fp32")
    scaler = torch.amp.GradScaler(device.type, enabled=args.dtype == "fp16")
    train_model = torch.compile(model) if args.compile else model

    bs = cfg.block_size
    tok_dtype = np.dtype(meta.get("dtype", "uint16"))  # uint32 for the Gemma-2 vocabulary
    train_loader = BinLoader(os.path.join(args.data_dir, "train.bin"), bs, args.batch_size, device, tok_dtype)
    val_loaders = {}
    for name in ("val", "val_ja", "val_en"):
        p = os.path.join(args.data_dir, f"{name}.bin")
        if os.path.exists(p) and os.path.getsize(p) // tok_dtype.itemsize > bs + 1:
            val_loaders[name] = BinLoader(p, bs, args.batch_size, device, tok_dtype)
    tokens_per_step = args.batch_size * args.grad_accum * bs
    print(f"Training data: {train_loader.n_tokens:,} tokens / {tokens_per_step:,} tokens per step"
          f" / {args.max_steps} steps = {tokens_per_step * args.max_steps / train_loader.n_tokens:.2f} epochs")

    log_f = open(os.path.join(args.out_dir, "train_log.jsonl"), "a", encoding="utf-8")
    step = start_step
    t0 = time.time()
    acc_loss = acc_aux = 0.0
    acc_counts = torch.zeros(cfg.n_layer, cfg.n_expert, device=device)
    x, y = train_loader.get_batch()
    model.train()
    try:
        while step < args.max_steps:
            lr = get_lr(step, args)
            for g in optimizer.param_groups:
                g["lr"] = lr

            for _ in range(args.grad_accum):
                with ctx:
                    out = train_model(x, y)
                    loss = out["loss"] / args.grad_accum
                x, y = train_loader.get_batch()  # prepare the next batch while the GPU computes
                scaler.scale(loss).backward()
                acc_loss += out["ce"].detach()
                acc_aux += out["aux"].detach()
                acc_counts += out["counts"]
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_interval == 0 or step == start_step + 1:
                n = args.grad_accum * (args.log_interval if step % args.log_interval == 0 else 1)
                dt = time.time() - t0
                tok_s = n / args.grad_accum * tokens_per_step / dt
                eta_h = (args.max_steps - step) * tokens_per_step / tok_s / 3600
                l, a = (acc_loss / n).item(), (acc_aux / n).item()
                util = acc_counts.sum(0).cpu()
                print(f"step {step:6d} | loss {l:.4f} | aux {a:.3f} | lr {lr:.2e} | {tok_s / 1e3:6.1f}k tok/s"
                      f" | ETA {eta_h:5.2f}h | expert% [{fmt_util(util)}]")
                log_f.write(json.dumps({"step": step, "loss": l, "aux": a, "lr": lr, "tok_s": tok_s,
                                        "expert_util": (util / util.sum()).tolist()}) + "\n")
                log_f.flush()
                acc_loss = acc_aux = 0.0
                acc_counts.zero_()
                t0 = time.time()

            if val_loaders and (step % args.eval_interval == 0 or step == args.max_steps):
                res = evaluate(model, val_loaders, args, ctx)
                print("  ── eval " + " | ".join(f"{k} loss {v['loss']:.4f} (ppl {math.exp(v['loss']):.1f})"
                                               for k, v in res.items()))
                if "val_ja" in res and "val_en" in res:
                    print("     Per-layer expert utilization (%)  ja | en")
                    for li in range(cfg.n_layer):
                        print(f"     L{li:<2d} ja [{fmt_util(res['val_ja']['counts'][li])}]"
                              f" | en [{fmt_util(res['val_en']['counts'][li])}]")
                val_loss = res.get("val", next(iter(res.values())))["loss"]
                log_f.write(json.dumps({"step": step, **{f"{k}_loss": v["loss"] for k, v in res.items()}}) + "\n")
                log_f.flush()
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(os.path.join(args.out_dir, "ckpt_best.pt"), model, optimizer, step, best_val, args)
                save_ckpt(ckpt_path, model, optimizer, step, best_val, args)
                print(f"  ── checkpoint saved: {ckpt_path} (best val {best_val:.4f})")
                t0 = time.time()

            if step % args.sample_interval == 0:
                print(f"  ── generation samples at step {step}")
                sample(model, sp, args.prompts, device, ctx, args.sample_tokens)
                t0 = time.time()
    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint...")
    save_ckpt(ckpt_path, model, optimizer, step, best_val, args)
    log_f.close()
    print(f"Saved: {ckpt_path} (step {step})")


if __name__ == "__main__":
    main()
