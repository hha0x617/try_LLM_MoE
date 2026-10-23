#!/bin/bash
# Japanese/English Wikipedia MoE model pipeline for macOS (download -> tokenizer -> data -> train -> GGUF)
# macOS version of ../run_pipeline.ps1. Training uses macosx/train.py (Apple GPU / MPS).
#
#   ./macosx/run_pipeline.sh                     # Full settings (small preset)
#   ./macosx/run_pipeline.sh --smoke             # Quick sanity check that finishes in a few minutes
#   ./macosx/run_pipeline.sh --skip-to train     # Resume from an intermediate step (download|tokenizer|data|train|export)
#   ./macosx/run_pipeline.sh --gemma-vocab       # Borrow the vocabulary from Gemma-2 (output to data_gemma/ / out_gemma/)
#   ./macosx/run_pipeline.sh --batch-size 4 --grad-accum 16   # Smaller micro-batch for Macs with less memory
#
# Files already built (Parquet, tokenizer.model, train.bin) are reused automatically, not rebuilt.
# Set PYTHON to use a Python other than .venv/bin/python.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1   # Flush output incrementally even when saving to a file (e.g. via tee)

SMOKE=0; GEMMA=0; SKIP_TO=download; PRESET=small; MAX_STEPS=20000; EN_SHARDS=8
JA_TOKENS=5e8; EN_TOKENS=5e8; OUTTYPE=f16; BATCH_SIZE=""; GRAD_ACCUM=""

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --smoke) SMOKE=1 ;;
        --gemma-vocab) GEMMA=1 ;;
        --skip-to) SKIP_TO="$2"; shift ;;
        --preset) PRESET="$2"; shift ;;
        --max-steps) MAX_STEPS="$2"; shift ;;
        --en-shards) EN_SHARDS="$2"; shift ;;   # Number of English shards to fetch (1 shard ≈ 280MB / ~100M tokens)
        --ja-tokens) JA_TOKENS="$2"; shift ;;
        --en-tokens) EN_TOKENS="$2"; shift ;;
        --outtype) OUTTYPE="$2"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift ;;
        --grad-accum) GRAD_ACCUM="$2"; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
    shift
done

case "$SKIP_TO" in
    download) START=0 ;; tokenizer) START=1 ;; data) START=2 ;; train) START=3 ;; export) START=4 ;;
    *) echo "--skip-to must be one of download|tokenizer|data|train|export" >&2; exit 1 ;;
esac

TOK_ARGS=(); TRAIN_ARGS=()
if [ "$SMOKE" = 1 ]; then
    DATA_DIR=smoke/data; OUT_DIR=smoke/out
    PRESET=tiny; MAX_STEPS=300; EN_SHARDS=1
    JA_TOKENS=3e6; EN_TOKENS=3e6
    TOK_ARGS=(--vocab_size 8000 --ja_docs 5000 --en_docs 5000)
    TRAIN_ARGS=(--batch_size 16 --grad_accum 2 --lr 2e-3 --warmup_steps 30
                --log_interval 50 --eval_interval 150 --sample_interval 300)
else
    DATA_DIR=data; OUT_DIR=out
fi
if [ "$GEMMA" = 1 ]; then
    # Output to separate folders so files don't mix with those built from our own vocabulary
    DATA_DIR="${DATA_DIR}_gemma"; OUT_DIR="${OUT_DIR}_gemma"
    # The 256K vocabulary makes the output layer 8x larger, so lower the micro-batch but keep the effective batch (64)
    if [ "$SMOKE" = 0 ]; then TRAIN_ARGS=(--batch_size 2 --grad_accum 32); fi
fi
# Explicit --batch-size / --grad-accum override the defaults above (argparse uses the last value)
if [ -n "$BATCH_SIZE" ]; then TRAIN_ARGS+=(--batch_size "$BATCH_SIZE"); fi
if [ -n "$GRAD_ACCUM" ]; then TRAIN_ARGS+=(--grad_accum "$GRAD_ACCUM"); fi

run() {
    "$PY" "$@" || { echo "Failed: python $*" >&2; exit 1; }
}
header() {
    printf '\n\033[36m=== %s ===\033[0m\n' "$1"
}

if [ ! -x "$PY" ]; then echo ".venv not found: $PY" >&2; exit 1; fi

if [ "$START" -le 0 ]; then
    header "[0] Download Wikipedia"
    if [ "$SMOKE" = 1 ]; then JA_SHARDS=1; else JA_SHARDS=0; fi
    run download_wiki.py --ja_shards "$JA_SHARDS" --en_shards "$EN_SHARDS"
fi
if [ "$START" -le 1 ]; then
    header "[1] Train the vocabulary"
    if [ -f "$DATA_DIR/tokenizer.model" ]; then echo "Already built, reusing: $DATA_DIR/tokenizer.model"
    elif [ "$GEMMA" = 1 ]; then run borrow_tokenizer.py --out_dir "$DATA_DIR"
    else run prepare_tokenizer.py --out_dir "$DATA_DIR" ${TOK_ARGS[@]+"${TOK_ARGS[@]}"}
    fi
fi
if [ "$START" -le 2 ]; then
    header "[2] Build the mixed Japanese/English binaries"
    if [ -f "$DATA_DIR/train.bin" ]; then echo "Already built, reusing: $DATA_DIR/train.bin"
    else run prepare_data.py --out_dir "$DATA_DIR" --tokenizer "$DATA_DIR/tokenizer.model" \
                 --ja_tokens "$JA_TOKENS" --en_tokens "$EN_TOKENS"
    fi
fi
if [ "$START" -le 3 ]; then
    header "[3] Train the MoE model"
    RESUME=(); if [ -f "$OUT_DIR/ckpt.pt" ]; then RESUME=(--resume); fi
    run macosx/train.py --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" --preset "$PRESET" \
        --max_steps "$MAX_STEPS" ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} ${RESUME[@]+"${RESUME[@]}"}
fi
if [ "$START" -le 4 ]; then
    header "[4] Convert to GGUF and verify"
    GGUF="$OUT_DIR/wiki-moe-ja-en-$OUTTYPE.gguf"
    run export_gguf.py --ckpt "$OUT_DIR/ckpt_best.pt" --tokenizer "$DATA_DIR/tokenizer.model" \
        --outtype "$OUTTYPE" --output "$GGUF"
    run verify_gguf.py "$GGUF" --ckpt "$OUT_DIR/ckpt_best.pt" --tokenizer "$DATA_DIR/tokenizer.model"
    printf '\n\033[32mDone: %s\033[0m\n' "$GGUF"
fi
