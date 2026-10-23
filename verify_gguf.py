"""
Step 5: Verify the contents of the exported GGUF file.

  python verify_gguf.py out/wiki-moe-ja-en-f16.gguf --ckpt out/ckpt_best.pt
  python verify_gguf.py out/wiki-moe-ja-en-f16.gguf --llama_cpp D:/tools/llama.cpp

1. Show the metadata and the tensor list (name, shape, type)
2. Rebuild a PyTorch model from the GGUF tensors alone and compare its output (logits) with the original checkpoint
   -> confirms that tensor names, shapes and ordering were written correctly
3. With --llama_cpp, actually run inference with llama.cpp (llama-simple / llama-completion) and
   show it side by side with PyTorch's greedy generation
"""
import argparse
import glob
import os
import subprocess

import gguf
import numpy as np
import sentencepiece as spm
import torch

from moe_model import MoEConfig, MoETransformer, load_checkpoint

# GGUF name -> PyTorch state_dict name
def gguf_to_torch_name(name):
    fixed = {"token_embd.weight": "tok_embeddings.weight", "output_norm.weight": "norm.weight",
             "output.weight": "output.weight"}
    if name in fixed:
        return fixed[name]
    _, i, rest = name.split(".", 2)
    table = {"attn_norm.weight": "attn_norm.weight", "attn_q.weight": "attn.wq.weight",
             "attn_k.weight": "attn.wk.weight", "attn_v.weight": "attn.wv.weight",
             "attn_output.weight": "attn.wo.weight", "ffn_norm.weight": "ffn_norm.weight",
             "ffn_gate_inp.weight": "moe.router.weight", "ffn_gate_exps.weight": "moe.w_gate",
             "ffn_up_exps.weight": "moe.w_up", "ffn_down_exps.weight": "moe.w_down"}
    return f"layers.{i}.{table[rest]}"


def field(reader, key, default=None):
    f = reader.fields.get(key)
    return f.contents() if f is not None else default


def load_gguf_model(path):
    reader = gguf.GGUFReader(path)
    arch = field(reader, "general.architecture")
    assert arch == "llama", f"Unexpected architecture: {arch}"
    k = lambda s: field(reader, f"llama.{s}")
    names = {t.name for t in reader.tensors}
    cfg = MoEConfig(
        vocab_size=int(k("vocab_size")), block_size=int(k("context_length")), n_layer=int(k("block_count")),
        n_head=int(k("attention.head_count")), n_kv_head=int(k("attention.head_count_kv")),
        d_model=int(k("embedding_length")), d_ff=int(k("feed_forward_length")),
        n_expert=int(k("expert_count")), n_expert_used=int(k("expert_used_count")),
        rope_theta=float(k("rope.freq_base")), norm_eps=float(k("attention.layer_norm_rms_epsilon")),
        tie_embeddings="output.weight" not in names,
    )
    model = MoETransformer(cfg)
    sd = {}
    for t in reader.tensors:
        arr = gguf.quants.dequantize(t.data, t.tensor_type)
        sd[gguf_to_torch_name(t.name)] = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    if cfg.tie_embeddings:
        sd["output.weight"] = sd["tok_embeddings.weight"]
    model.load_state_dict(sd)
    return reader, model


def print_summary(reader):
    print("=== Metadata ===")
    for key, f in reader.fields.items():
        if key.startswith("tokenizer.ggml.") and key.split(".")[-1] in ("tokens", "scores", "token_type", "merges"):
            print(f"  {key:40s} [{len(f.data)} elements]")
        elif not key.startswith("GGUF."):
            print(f"  {key:40s} {f.contents()}")
    print("\n=== Tensors (shapes in ggml order = reverse of PyTorch) ===")
    total = 0
    for t in reader.tensors:
        total += int(t.n_elements)
        print(f"  {t.name:28s} {str(t.shape.tolist()):22s} {t.tensor_type.name}")
    print(f"  Total {len(reader.tensors)} tensors / {total / 1e6:.1f}M parameters")


def find_llama_bin(llama_dir):
    for exe in ("llama-simple", "llama-completion"):
        hits = glob.glob(os.path.join(llama_dir, "**", exe + ("*.exe" if os.name == "nt" else "")), recursive=True)
        if hits:
            return exe, hits[0]
    return None, None


def main():
    ap = argparse.ArgumentParser(description="Verify a GGUF file")
    ap.add_argument("gguf_path")
    ap.add_argument("--ckpt", default=None, help="Original checkpoint to compare against")
    ap.add_argument("--tokenizer", default=os.path.join("data", "tokenizer.model"))
    ap.add_argument("--prompt", default="日本の首都は")
    ap.add_argument("--n_predict", type=int, default=24)
    ap.add_argument("--llama_cpp", default=None, help="llama.cpp build/release directory")
    args = ap.parse_args()

    reader, gmodel = load_gguf_model(args.gguf_path)
    print_summary(reader)
    gmodel.eval()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    ids = [sp.bos_id()] + sp.encode(args.prompt)
    x = torch.tensor([ids])

    print("\n=== PyTorch (model with weights read back from the GGUF) ===")
    with torch.no_grad():
        g_logits = gmodel(x)["logits"]
        out = gmodel.generate(x, args.n_predict, temperature=0)
    print(f"  greedy: {sp.decode(out[0].tolist()[1:])!r}")

    if args.ckpt:
        cmodel, _ = load_checkpoint(args.ckpt, "cpu")
        cmodel.eval()
        with torch.no_grad():
            c_logits = cmodel(x)["logits"]
        diff = (g_logits - c_logits).abs().max().item()
        same = (g_logits.argmax(-1) == c_logits.argmax(-1)).float().mean().item()
        print(f"  vs. original checkpoint: max logits error {diff:.4g} / next-token prediction agreement {same * 100:.1f}%")

    if args.llama_cpp:
        exe, path = find_llama_bin(args.llama_cpp)
        if not path:
            print(f"\nllama-simple / llama-completion not found in {args.llama_cpp}")
            return
        print(f"\n=== llama.cpp ({os.path.basename(path)}) ===")
        if exe == "llama-simple":
            cmd = [path, "-m", args.gguf_path, "-n", str(args.n_predict), args.prompt]
        else:
            cmd = [path, "-m", args.gguf_path, "-n", str(args.n_predict), "-p", args.prompt,
                   "--temp", "0", "-no-cnv", "--no-display-prompt"]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(r.stdout.strip() or r.stderr[-3000:])


if __name__ == "__main__":
    main()
