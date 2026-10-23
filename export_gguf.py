"""
Step 4: Bundle the trained checkpoint (.pt) and the tokenizer into a single GGUF file.

  python export_gguf.py --ckpt out/ckpt_best.pt --outtype f16 --output out/wiki-moe-ja-en-f16.gguf

The GGUF architecture is written as "llama" (which becomes a Mixtral-style MoE when expert_count > 0),
so existing inference engines such as llama.cpp / Ollama can load it as-is.

Tensor name mapping (N = layer index):
  tok_embeddings.weight          -> token_embd.weight         [vocab, d_model]
  layers.N.attn_norm.weight      -> blk.N.attn_norm.weight    [d_model]
  layers.N.attn.wq / wk / wv / wo-> blk.N.attn_q / attn_k / attn_v / attn_output.weight
  layers.N.ffn_norm.weight       -> blk.N.ffn_norm.weight
  layers.N.moe.router.weight     -> blk.N.ffn_gate_inp.weight [n_expert, d_model]
  layers.N.moe.w_gate            -> blk.N.ffn_gate_exps.weight[n_expert, d_ff, d_model]  (3-D)
  layers.N.moe.w_up              -> blk.N.ffn_up_exps.weight  [n_expert, d_ff, d_model]  (3-D)
  layers.N.moe.w_down            -> blk.N.ffn_down_exps.weight[n_expert, d_model, d_ff]  (3-D)
  norm.weight                    -> output_norm.weight
  output.weight                  -> output.weight             (omitted when weights are tied = token_embd is reused)
* Shapes are in PyTorch (numpy) order. Inside GGUF (ggml) the dimensions are recorded in reverse order.
* RoPE is trained with the same adjacent-pair scheme as llama.cpp, so no Q/K permutation (as in HF conversion) is needed.
"""
import argparse
import os

import numpy as np
import sentencepiece as spm
import torch
import gguf

from moe_model import load_checkpoint

QTYPES = {
    "f32": (gguf.GGMLQuantizationType.F32, gguf.LlamaFileType.ALL_F32),
    "f16": (gguf.GGMLQuantizationType.F16, gguf.LlamaFileType.MOSTLY_F16),
    "bf16": (gguf.GGMLQuantizationType.BF16, gguf.LlamaFileType.MOSTLY_BF16),
    "q8_0": (gguf.GGMLQuantizationType.Q8_0, gguf.LlamaFileType.MOSTLY_Q8_0),
}


def tensor_map(model):
    """Yield (GGUF name, PyTorch tensor) pairs."""
    cfg = model.cfg
    yield "token_embd.weight", model.tok_embeddings.weight
    for i, layer in enumerate(model.layers):
        b = f"blk.{i}"
        yield f"{b}.attn_norm.weight", layer.attn_norm.weight
        yield f"{b}.attn_q.weight", layer.attn.wq.weight
        yield f"{b}.attn_k.weight", layer.attn.wk.weight
        yield f"{b}.attn_v.weight", layer.attn.wv.weight
        yield f"{b}.attn_output.weight", layer.attn.wo.weight
        yield f"{b}.ffn_norm.weight", layer.ffn_norm.weight
        yield f"{b}.ffn_gate_inp.weight", layer.moe.router.weight
        yield f"{b}.ffn_gate_exps.weight", layer.moe.w_gate
        yield f"{b}.ffn_up_exps.weight", layer.moe.w_up
        yield f"{b}.ffn_down_exps.weight", layer.moe.w_down
    yield "output_norm.weight", model.norm.weight
    if not cfg.tie_embeddings:
        yield "output.weight", model.output.weight


def choose_qtype(name, data, qtype):
    # Keep 1-D tensors (RMSNorm) and the router in F32 to preserve precision (same policy as llama.cpp's conversion script)
    if data.ndim == 1 or name.endswith("ffn_gate_inp.weight"):
        return gguf.GGMLQuantizationType.F32
    if qtype == gguf.GGMLQuantizationType.Q8_0 and data.shape[-1] % 32 != 0:
        return gguf.GGMLQuantizationType.F16  # Q8_0 requires the row length to be a multiple of 32
    return qtype


def add_tokenizer(writer, sp, vocab_size):
    tokens, scores, types = [], [], []
    for i in range(sp.vocab_size()):
        tokens.append(sp.id_to_piece(i).encode("utf-8"))
        scores.append(sp.get_score(i))
        if sp.is_unknown(i):
            types.append(gguf.TokenType.UNKNOWN)
        elif sp.is_control(i):
            types.append(gguf.TokenType.CONTROL)
        elif sp.is_unused(i):
            types.append(gguf.TokenType.UNUSED)
        elif sp.is_byte(i):
            types.append(gguf.TokenType.BYTE)
        else:
            types.append(gguf.TokenType.NORMAL)
    # If the model's vocab size exceeds the tokenizer's, pad with unused tokens
    for i in range(sp.vocab_size(), vocab_size):
        tokens.append(f"[PAD{i}]".encode("utf-8"))
        scores.append(-1000.0)
        types.append(gguf.TokenType.UNUSED)

    writer.add_tokenizer_model("llama")  # SentencePiece (SPM) format
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_bos_token_id(sp.bos_id())
    writer.add_eos_token_id(sp.eos_id())
    writer.add_unk_token_id(sp.unk_id())
    if sp.pad_id() >= 0:
        writer.add_pad_token_id(sp.pad_id())
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)
    # Match SentencePiece's add_dummy_prefix (True for our own vocabulary, False for Gemma-2)
    writer.add_add_space_prefix(sp.encode("a", out_type=str)[0].startswith("▁"))


def main():
    ap = argparse.ArgumentParser(description="Convert a trained MoE model to GGUF")
    ap.add_argument("--ckpt", default=os.path.join("out", "ckpt_best.pt"))
    ap.add_argument("--tokenizer", default=os.path.join("data", "tokenizer.model"))
    ap.add_argument("--output", default=None, help="Default: out/wiki-moe-ja-en-<outtype>.gguf")
    ap.add_argument("--outtype", default="f16", choices=list(QTYPES),
                    help="Type of the weight matrices (for further size reduction, use llama.cpp's "
                         "llama-quantize to convert to Q4_K_M etc.)")
    ap.add_argument("--name", default="wiki-moe-ja-en")
    args = ap.parse_args()
    output = args.output or os.path.join("out", f"{args.name}-{args.outtype}.gguf")

    model, ckpt = load_checkpoint(args.ckpt, "cpu")
    model.eval()
    cfg = model.cfg
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    if sp.vocab_size() > cfg.vocab_size:
        raise SystemExit(f"Tokenizer vocabulary size {sp.vocab_size()} exceeds the model's vocabulary size {cfg.vocab_size}")
    qtype, ftype = QTYPES[args.outtype]
    total, active = model.num_params()
    print(f"Checkpoint: {args.ckpt} (step {ckpt.get('step')})")
    print(f"Model: {cfg}")

    writer = gguf.GGUFWriter(output, "llama")
    writer.add_name(args.name)
    writer.add_description(f"Mixture-of-Experts LM trained from scratch on Japanese/English Wikipedia "
                           f"({total / 1e6:.0f}M total / {active / 1e6:.0f}M active params, step {ckpt.get('step')})")
    writer.add_vocab_size(cfg.vocab_size)
    writer.add_context_length(cfg.block_size)
    writer.add_embedding_length(cfg.d_model)
    writer.add_block_count(cfg.n_layer)
    writer.add_feed_forward_length(cfg.d_ff)
    writer.add_head_count(cfg.n_head)
    writer.add_head_count_kv(cfg.n_kv_head)
    writer.add_rope_dimension_count(cfg.head_dim)
    writer.add_rope_freq_base(cfg.rope_theta)
    writer.add_layer_norm_rms_eps(cfg.norm_eps)
    writer.add_expert_count(cfg.n_expert)            # llama.expert_count
    writer.add_expert_used_count(cfg.n_expert_used)  # llama.expert_used_count (Top-k)
    writer.add_file_type(ftype)
    add_tokenizer(writer, sp, cfg.vocab_size)

    with torch.no_grad():
        for name, t in tensor_map(model):
            data = t.detach().float().contiguous().numpy()
            tq = choose_qtype(name, data, qtype)
            if tq == gguf.GGMLQuantizationType.F32:
                writer.add_tensor(name, data)
            else:
                writer.add_tensor(name, gguf.quants.quantize(data, tq), raw_dtype=tq)
            print(f"  {name:28s} {str(list(data.shape)):22s} {tq.name}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"\nGGUF written: {output} ({os.path.getsize(output) / 1024 ** 2:,.1f} MB)")


if __name__ == "__main__":
    main()
