"""
Step 1: Train a SentencePiece (BPE) vocabulary on Japanese/English Wikipedia.

Outputs:
  data/tokenizer.model   ... used by the training script and the GGUF export script
  data/tokenizer.vocab   ... human-readable vocabulary list

Settings are chosen to behave the same as llama.cpp's SPM tokenizer ("llama" type).
  - normalization: identity (llama.cpp does no NFKC normalization, so training and inference tokenization match)
  - byte_fallback: characters not in the vocabulary are split into <0xXX> byte tokens (no unknown tokens)
  - add_dummy_prefix: prepend a space (corresponds to add_space_prefix=true in GGUF)
"""
import argparse
import os

import sentencepiece as spm
from tqdm import tqdm

from wiki_common import DEFAULT_DUMP, DEFAULT_RAW_DIR, stream_articles


def dump_corpus(path, ja_docs, en_docs, dump, seed, max_line_chars, raw_dir):
    n_lines = 0
    with open(path, "w", encoding="utf-8") as f:
        for lang, n_docs in (("ja", ja_docs), ("en", en_docs)):
            if n_docs <= 0:
                continue
            it = stream_articles(lang, dump=dump, seed=seed, raw_dir=raw_dir)
            for _ in tqdm(range(n_docs), desc=f"corpus[{lang}]"):
                try:
                    text = next(it)
                except StopIteration:
                    break
                # SentencePiece treats 1 line = 1 sentence, so write out one paragraph per line
                for line in text.split("\n"):
                    line = line.strip()
                    if line:
                        f.write(line[:max_line_chars] + "\n")
                        n_lines += 1
    return n_lines


def main():
    ap = argparse.ArgumentParser(description="Train a SentencePiece vocabulary on Japanese/English Wikipedia")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--ja_docs", type=int, default=150_000, help="number of Japanese articles to train on")
    ap.add_argument("--en_docs", type=int, default=150_000, help="number of English articles to train on")
    ap.add_argument("--dump", default=DEFAULT_DUMP, help="Wikipedia dump date (e.g. 20231101)")
    ap.add_argument("--raw_dir", default=DEFAULT_RAW_DIR, help="download_wiki.py output directory (streams if absent)")
    ap.add_argument("--input_sentence_size", type=int, default=3_000_000, help="max number of lines SentencePiece actually uses")
    ap.add_argument("--max_line_chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1234, help="defaults to a different seed so the article sample differs from the training data")
    ap.add_argument("--keep_corpus", action="store_true", help="do not delete the intermediate text corpus")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    corpus = os.path.join(args.out_dir, "tokenizer_corpus.txt")
    n_lines = dump_corpus(corpus, args.ja_docs, args.en_docs, args.dump, args.seed, args.max_line_chars, args.raw_dir)
    print(f"Corpus written: {n_lines:,} lines -> {corpus}")

    prefix = os.path.join(args.out_dir, "tokenizer")
    spm.SentencePieceTrainer.train(
        input=corpus,
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=0.9995,         # enough to cover Japanese kanji
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        add_dummy_prefix=True,
        allow_whitespace_only_pieces=True,
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1,  # same ID assignment as Llama
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,
        max_sentence_length=16384,
        num_threads=os.cpu_count(),
        train_extremely_large_corpus=False,
    )
    if not args.keep_corpus:
        os.remove(corpus)

    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    print(f"\nVocabulary created: {prefix}.model (vocab size {sp.vocab_size():,})")
    for s in ["日本の首都は東京である。", "The capital of Japan is Tokyo."]:
        print(f"  {s!r} -> {sp.encode(s, out_type=str)}")


if __name__ == "__main__":
    main()
