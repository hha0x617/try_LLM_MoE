"""
Step 2: Tokenize Japanese/English Wikipedia and build the "mixed Japanese/English binaries".

Outputs (all headerless raw binaries: a flat sequence of uint16 token IDs (uint32 if the vocabulary exceeds 65,536)):
  data/train.bin      ... training (Japanese and English articles interleaved, article by article)
  data/val.bin        ... validation (mixed Japanese/English)
  data/val_ja.bin     ... validation (Japanese only; for per-language loss / expert utilization)
  data/val_en.bin     ... validation (English only)
  data/meta.json      ... dtype, vocabulary size, per-language token counts, etc.

Vocabulary size 32000 < 65536, so tokens are stored as uint16 (2 bytes), half the size of uint32.
With the Gemma-2 vocabulary (256,000 tokens, borrow_tokenizer.py) uint32 is used automatically.
Each article is concatenated as [BOS] text [EOS].

Existing .bin files can be reused as-is (no need to rebuild them with the same settings).
"""
import argparse
import json
import os
import random

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

from wiki_common import DEFAULT_DUMP, DEFAULT_RAW_DIR, stream_articles

CHUNK_DOCS = 256  # articles tokenized per batch (to make use of SentencePiece multithreading)


class LangSource:
    def __init__(self, lang, target_tokens, args):
        self.lang = lang
        self.target = int(target_tokens)
        self.written = 0
        self.docs = 0
        self.it = stream_articles(lang, dump=args.dump, seed=args.seed, raw_dir=args.raw_dir)
        self.exhausted = False

    @property
    def done(self):
        return self.exhausted or self.written >= self.target

    @property
    def progress(self):
        return self.written / max(self.target, 1)

    def next_chunk(self):
        texts = []
        for _ in range(CHUNK_DOCS):
            try:
                texts.append(next(self.it))
            except StopIteration:
                self.exhausted = True
                break
        return texts


def main():
    ap = argparse.ArgumentParser(description="Build mixed Japanese/English training binaries from Wikipedia")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--tokenizer", default=os.path.join("data", "tokenizer.model"))
    ap.add_argument("--ja_tokens", type=float, default=500e6, help="target number of Japanese tokens (0 = no Japanese)")
    ap.add_argument("--en_tokens", type=float, default=500e6, help="target number of English tokens (0 = no English)")
    ap.add_argument("--val_ratio", type=float, default=0.005, help="fraction of articles held out for validation")
    ap.add_argument("--dump", default=DEFAULT_DUMP)
    ap.add_argument("--raw_dir", default=DEFAULT_RAW_DIR, help="download_wiki.py output directory (streams if absent)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true", help="overwrite an existing train.bin")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.bin")
    if os.path.exists(train_path) and not args.overwrite:
        raise SystemExit(f"{train_path} already exists and can be reused as is (use --overwrite to rebuild)")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    bos, eos, vocab = sp.bos_id(), sp.eos_id(), sp.vocab_size()
    dtype = np.uint16 if vocab <= 65536 else np.uint32

    sources = [LangSource(l, t, args) for l, t in (("ja", args.ja_tokens), ("en", args.en_tokens)) if t > 0]
    rng = random.Random(args.seed)
    files = {name: open(os.path.join(args.out_dir, f"{name}.bin"), "wb")
             for name in ["train", "val"] + [f"val_{s.lang}" for s in sources]}
    counts = {name: 0 for name in files}

    total_target = sum(s.target for s in sources)
    pbar = tqdm(total=total_target, unit="tok", unit_scale=True, desc="tokenize")
    try:
        while True:
            active = [s for s in sources if not s.done]
            if not active:
                break
            # pull from the language furthest behind its target -> ja/en mix at the target ratio across the file
            src = min(active, key=lambda s: s.progress)
            texts = src.next_chunk()
            if not texts:
                continue
            for ids in sp.encode(texts, num_threads=os.cpu_count()):
                arr = np.array([bos] + ids + [eos], dtype=dtype)
                if rng.random() < args.val_ratio:
                    for name in ("val", f"val_{src.lang}"):
                        files[name].write(arr.tobytes())
                        counts[name] += len(arr)
                else:
                    files["train"].write(arr.tobytes())
                    counts["train"] += len(arr)
                    src.written += len(arr)
                    pbar.update(len(arr))
                src.docs += 1
            pbar.set_postfix({s.lang: f"{s.written / 1e6:.0f}M" for s in sources})
    finally:
        pbar.close()
        for f in files.values():
            f.close()

    meta = {
        "dtype": np.dtype(dtype).name,
        "vocab_size": vocab,
        "bos_id": bos,
        "eos_id": eos,
        "tokenizer": os.path.abspath(args.tokenizer),
        "dump": args.dump,
        "tokens": counts,
        "train_tokens_by_lang": {s.lang: s.written for s in sources},
        "docs_by_lang": {s.lang: s.docs for s in sources},
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nDone:")
    for name, n in counts.items():
        print(f"  {name}.bin : {n:,} tokens ({n * 2 / 1024 ** 2:,.1f} MB)")
    for s in sources:
        note = " (data exhausted)" if s.exhausted and s.written < s.target else ""
        print(f"  [{s.lang}] {s.docs:,} articles / {s.written:,} train tokens{note}")


if __name__ == "__main__":
    main()
