"""
Generate text from a trained checkpoint (PyTorch version). (macOS version of ../generate.py)

  python macosx/generate.py --prompt "日本の首都は" --prompt "The capital of France is"
  python macosx/generate.py --show_routing --prompt "東京タワーは Tokyo Tower とも呼ばれる"

Differences from ../generate.py:
  - uses the Apple GPU (MPS) when available, otherwise the CPU
  - imports macosx/moe_model.py (real-valued RoPE)
"""
import argparse
import os

import sentencepiece as spm
import torch

from moe_model import load_checkpoint


def show_routing(model, sp, text, device):
    """Show, per layer, which experts each token was routed to."""
    ids = [sp.bos_id()] + sp.encode(text)
    chosen = []
    hooks = [layer.moe.router.register_forward_hook(
        lambda m, i, o: chosen.append(o.float().softmax(-1).topk(model.cfg.n_expert_used, dim=-1).indices.cpu()))  # [N, k]
        for layer in model.layers]
    with torch.no_grad():
        model(torch.tensor([ids], device=device))
    for h in hooks:
        h.remove()
    print(f"\nRouting (top-{model.cfg.n_expert_used} experts per token, per layer)")
    print("token".ljust(14) + " ".join(f"L{i:<5d}" for i in range(len(chosen))))
    for t, tid in enumerate(ids):
        piece = sp.id_to_piece(tid)
        print(piece[:13].ljust(14) + " ".join(",".join(map(str, c[t].tolist())).ljust(6) for c in chosen))


def main():
    ap = argparse.ArgumentParser(description="Generate text with a trained MoE model")
    ap.add_argument("--ckpt", default=os.path.join("out", "ckpt_best.pt"))
    ap.add_argument("--tokenizer", default=os.path.join("data", "tokenizer.model"))
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy (deterministic)")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show_routing", action="store_true")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model, ckpt = load_checkpoint(args.ckpt, device)
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    print(f"Loaded {args.ckpt} (step {ckpt['step']})")

    for p in args.prompt or ["日本の首都は", "The capital of France is"]:
        idx = torch.tensor([[sp.bos_id()] + sp.encode(p)], device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k, eos_id=sp.eos_id())
        print(f"\n[{p}]\n{sp.decode(out[0].tolist()[1:])}")
        if args.show_routing:
            show_routing(model, sp, p, device)


if __name__ == "__main__":
    main()
