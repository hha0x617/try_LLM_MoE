"""
Alternative to step 1: borrow the vocabulary (tokenizer.model) that ships with Gemma-2.

  python borrow_tokenizer.py --out_dir data_gemma               # download from Hugging Face
  python borrow_tokenizer.py --out_dir data_gemma --file x.model  # use a local file

Outputs:
  <out_dir>/tokenizer.model   ... same location and name as the one prepare_tokenizer.py creates
  <out_dir>/tokenizer.vocab   ... human-readable vocabulary list (piece <TAB> score)

Gemma-2 is a gated model on Hugging Face, so two things are needed beforehand:
  1. Open https://huggingface.co/google/gemma-2-2b and accept the license (Gemma Terms of Use)
  2. Create an access token (Read permission) and log in with `hf auth login` (or --token / env var HF_TOKEN)
The vocabulary is shared by Gemma-2 2B / 9B / 27B, so any of those repositories gives the same file.

Differences from our own vocabulary (32,000 tokens):
  - 256,000 > 65,536 tokens, so prepare_data.py stores token sequences as uint32 (2x file size)
  - No leading space is added (add_dummy_prefix=False); export_gguf.py detects this and writes it to the GGUF
  - Special token IDs differ (pad=0, eos=1, bos=2, unk=3); scripts look them up via sp.bos_id() etc., so they work as-is
  - The output layer is 8x larger, so lower train.py to e.g. --batch_size 2 --grad_accum 32 (with 16GB VRAM)
"""
import argparse
import os
import shutil

import sentencepiece as spm

DEFAULT_REPO = "google/gemma-2-2b"


def download(repo, token):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    try:
        return hf_hub_download(repo, "tokenizer.model", token=token)
    except GatedRepoError:
        raise SystemExit(
            f"Could not download {repo} because it is a gated model.\n"
            f"  1. Accept the license at https://huggingface.co/{repo}\n"
            f"  2. Log in with `hf auth login` (or use --token / the HF_TOKEN environment variable)\n"
            f"Complete both steps, then run this again.")


def main():
    ap = argparse.ArgumentParser(description="Borrow the Gemma-2 vocabulary (tokenizer.model)")
    ap.add_argument("--out_dir", default="data_gemma", help="defaults to a location separate from your own data/ (to keep them apart)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--file", default=None, help="use an already downloaded tokenizer.model")
    ap.add_argument("--token", default=None, help="Hugging Face access token (default: login credentials / HF_TOKEN)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite an existing tokenizer.model")
    args = ap.parse_args()

    dst = os.path.join(args.out_dir, "tokenizer.model")
    if os.path.exists(dst) and not args.overwrite:
        raise SystemExit(f"{dst} already exists (use --overwrite to replace it)")

    src = args.file or download(args.repo, args.token)
    sp = spm.SentencePieceProcessor(model_file=src)  # fails here if the format is unreadable
    os.makedirs(args.out_dir, exist_ok=True)
    shutil.copyfile(src, dst)
    with open(os.path.join(args.out_dir, "tokenizer.vocab"), "w", encoding="utf-8") as f:
        for i in range(sp.vocab_size()):
            f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i):g}\n")

    n = sp.vocab_size()
    n_byte = sum(sp.is_byte(i) for i in range(n))
    n_ctrl = sum(sp.is_control(i) for i in range(n))
    dummy_prefix = sp.encode("a", out_type=str)[0].startswith("▁")
    print(f"Source: {src}")
    print(f"Vocabulary saved: {dst} (vocab size {n:,})")
    print(f"  Special tokens: unk={sp.unk_id()} bos={sp.bos_id()} eos={sp.eos_id()} pad={sp.pad_id()}"
          f" / {n_ctrl} control tokens / {n_byte} byte tokens")
    print(f"  Leading space (add_dummy_prefix): {dummy_prefix}")
    print(f"  Token storage dtype: {'uint16' if n <= 65536 else 'uint32'}")
    for s in ["日本の首都は東京である。", "The capital of Japan is Tokyo."]:
        print(f"  {s!r} -> {sp.encode(s, out_type=str)}")
    print(f"\nNext step: python prepare_data.py --out_dir {args.out_dir} --tokenizer {dst}")


if __name__ == "__main__":
    main()
