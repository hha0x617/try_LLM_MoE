"""
Step 0: Download the Wikipedia (Japanese / English) Parquet files and save them locally.

Once downloaded, later vocabulary training and training-data preparation
read from the local files without network access (detected automatically by wiki_common.stream_articles).

Location: data/raw/wikipedia/<dump>.<lang>/train-XXXXX-of-XXXXX.parquet
  20231101.ja : 15 files / about 3.9GB
  20231101.en : 41 files / about 11.6GB  (--en_shards fetches only some shards)

If interrupted, re-running skips files that were already downloaded.
"""
import argparse
import os

from huggingface_hub import HfApi, hf_hub_download

from wiki_common import DATASET, DEFAULT_DUMP, DEFAULT_RAW_DIR


def download_lang(api, raw_dir, dump, lang, max_shards):
    files = sorted(
        s.rfilename for s in api.dataset_info(DATASET, files_metadata=False).siblings
        if s.rfilename.startswith(f"{dump}.{lang}/") and s.rfilename.endswith(".parquet")
    )
    if not files:
        raise SystemExit(f"Error: {DATASET} does not contain {dump}.{lang}")
    if max_shards > 0:
        files = files[:max_shards]
    print(f"[{lang}] Downloading {len(files)} files to {raw_dir}")
    for i, name in enumerate(files, 1):
        dst = os.path.join(raw_dir, name)
        if os.path.exists(dst):
            print(f"  ({i}/{len(files)}) already downloaded: {name}")
            continue
        print(f"  ({i}/{len(files)}) downloading: {name}")
        hf_hub_download(DATASET, name, repo_type="dataset", local_dir=raw_dir)


def main():
    ap = argparse.ArgumentParser(description="Save Wikipedia Parquet files locally")
    ap.add_argument("--raw_dir", default=DEFAULT_RAW_DIR)
    ap.add_argument("--dump", default=DEFAULT_DUMP)
    ap.add_argument("--langs", nargs="+", default=["ja", "en"])
    ap.add_argument("--ja_shards", type=int, default=0, help="max number of Japanese shards (0 = all)")
    ap.add_argument("--en_shards", type=int, default=0, help="max number of English shards (0 = all, 1 shard ~ 280MB)")
    args = ap.parse_args()

    api = HfApi()
    for lang in args.langs:
        limit = {"ja": args.ja_shards, "en": args.en_shards}.get(lang, 0)
        download_lang(api, args.raw_dir, args.dump, lang, limit)
    print("Done. Subsequent scripts will use these local files automatically.")


if __name__ == "__main__":
    main()
