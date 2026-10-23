"""Shared code for reading Wikipedia (Japanese / English) articles.

Reads the local Parquet files saved by download_wiki.py if present;
otherwise uses streaming from Hugging Face (which downloads every time).
"""
import glob
import os
import random
import re

DATASET = "wikimedia/wikipedia"   # official Wikimedia dataset (plain text, wikitext markup removed)
DEFAULT_DUMP = "20231101"
DEFAULT_RAW_DIR = os.path.join("data", "raw", "wikipedia")

_MULTI_NL = re.compile(r"\n{3,}")


def local_files(lang, dump=DEFAULT_DUMP, raw_dir=DEFAULT_RAW_DIR):
    return sorted(glob.glob(os.path.join(raw_dir, f"{dump}.{lang}", "*.parquet")))


def _iter_local(files, seed, buffer_size):
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    files = files[:]
    rng.shuffle(files)  # shuffle file order
    buf = []
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1024, columns=["text"]):
            for text in batch.column(0).to_pylist():
                # buffer shuffle: yield one random item from the filled buffer
                if len(buf) < buffer_size:
                    buf.append(text)
                    continue
                j = rng.randrange(buffer_size)
                yield buf[j]
                buf[j] = text
    rng.shuffle(buf)
    yield from buf


def _iter_remote(lang, dump, seed, buffer_size):
    from datasets import load_dataset

    ds = load_dataset(DATASET, f"{dump}.{lang}", split="train", streaming=True)
    for item in ds.shuffle(seed=seed, buffer_size=buffer_size):
        yield item["text"]


def stream_articles(lang, dump=DEFAULT_DUMP, seed=42, buffer_size=10_000, min_chars=200,
                    raw_dir=DEFAULT_RAW_DIR, verbose=True):
    """Generator yielding article texts one at a time (shuffled)."""
    files = local_files(lang, dump, raw_dir)
    if files:
        if verbose:
            print(f"[{lang}] Using {len(files)} local Parquet files: {os.path.dirname(files[0])}")
        it = _iter_local(files, seed, buffer_size)
    else:
        if verbose:
            print(f"[{lang}] No local files found; streaming from Hugging Face"
                  f" (run download_wiki.py first to reuse the data)")
        it = _iter_remote(lang, dump, seed, buffer_size)
    for text in it:
        text = _MULTI_NL.sub("\n\n", text.strip())
        if len(text) >= min_chars:  # skip stub articles (extremely short ones)
            yield text
