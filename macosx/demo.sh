#!/bin/bash
# Live demo script for the presentation (runs the 7 steps of the "live demo" slides as-is)
# macOS version of ../demo.ps1. Training uses macosx/train.py (Apple GPU / MPS).
#
#   ./macosx/demo.sh                              # Advance one step at a time with Enter
#   ./macosx/demo.sh --reset                      # Delete the previous demo folder and Ollama model before starting
#   ./macosx/demo.sh --no-pause                   # Run straight through without pausing (for rehearsal)
#   ./macosx/demo.sh --start-at 4                 # Resume from an intermediate step
#   ./macosx/demo.sh --start-at 3 --max-steps 400 # Reuse the built data and try a different number of training steps
#   ./macosx/demo.sh --llama-cpp-dir ~/llama.cpp  # Location of llama-completion (searched on PATH if omitted)
#
# Prerequisite: Parquet files already downloaded to data/raw/wikipedia/ (download_wiki.py)
# Set PYTHON to use a Python other than .venv/bin/python.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONIOENCODING=utf-8

RESET=0; NO_PAUSE=0; START_AT=1; MAX_STEPS=200; LLAMA_CPP_DIR=""
PROMPT="日本の首都は"; OLLAMA_MODEL=demo-moe

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --reset) RESET=1 ;;
        --no-pause) NO_PAUSE=1 ;;
        --start-at) START_AT="$2"; shift ;;
        --max-steps) MAX_STEPS="$2"; shift ;;   # Number of training steps for DEMO 3
        --llama-cpp-dir) LLAMA_CPP_DIR="$2"; shift ;;
        --prompt) PROMPT="$2"; shift ;;
        --ollama-model) OLLAMA_MODEL="$2"; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
    shift
done
if ! [ "$START_AT" -ge 1 ] 2>/dev/null || [ "$START_AT" -gt 7 ]; then echo "--start-at must be 1-7" >&2; exit 1; fi
if ! [ "$MAX_STEPS" -ge 10 ] 2>/dev/null || [ "$MAX_STEPS" -gt 100000 ]; then echo "--max-steps must be 10-100000" >&2; exit 1; fi

DATA_DIR=demo/data
OUT_DIR=demo/out
GGUF="$OUT_DIR/demo-moe-f16.gguf"

GRAY='\033[90m'; CYAN='\033[36m'; YELLOW='\033[33m'; RED='\033[31m'; GREEN='\033[32m'; RESET_C='\033[0m'
LINE="$(printf '=%.0s' $(seq 1 70))"

step() {  # step <n> <title> : returns non-zero if the step is skipped
    [ "$1" -lt "$START_AT" ] && return 1
    printf "\n${GRAY}%s${RESET_C}\n${CYAN} DEMO %s / 7 : %s${RESET_C}\n${GRAY}%s${RESET_C}\n" "$LINE" "$1" "$2" "$LINE"
}
show() {
    printf "${YELLOW}\$ %s${RESET_C}\n" "$*"
}
run() {
    show python "$@"
    local t0; t0=$(date +%s)
    "$PY" "$@" || { echo "Failed: python $*" >&2; exit 1; }
    printf "${GRAY}  (%d s)${RESET_C}\n" $(( $(date +%s) - t0 ))
}
wait_next() {
    if [ "$NO_PAUSE" = 0 ]; then printf '\n[Enter] to continue'; read -r _; fi
}

# --- Preflight checks ---
if [ ! -x "$PY" ]; then echo ".venv not found: $PY" >&2; exit 1; fi
if ! ls data/raw/wikipedia/*/*.parquet >/dev/null 2>&1; then
    echo "No Parquet files in data/raw/wikipedia. Run download_wiki.py --ja_shards 1 --en_shards 1 first" >&2; exit 1
fi
LLAMA=""
if [ -n "$LLAMA_CPP_DIR" ]; then
    LLAMA="$(find "$LLAMA_CPP_DIR" -type f -name llama-completion -perm -u+x 2>/dev/null | head -n 1)"
else
    LLAMA="$(command -v llama-completion || true)"
fi
OLLAMA="$(command -v ollama || true)"
echo "llama.cpp : ${LLAMA:-not found (skipping DEMO 6)}"
echo "Ollama    : ${OLLAMA:-not found (skipping DEMO 7)}"

if [ "$RESET" = 1 ]; then
    if [ -d demo ]; then rm -rf demo; echo "Deleted the demo folder"; fi
    if [ -n "$OLLAMA" ]; then "$OLLAMA" rm "$OLLAMA_MODEL" >/dev/null 2>&1; echo "Removed Ollama model $OLLAMA_MODEL (if registered)"; fi
fi
mkdir -p "$DATA_DIR" "$OUT_DIR"

# --- DEMO 1: Vocabulary ---
if step 1 "Train the vocabulary (SentencePiece)  ── about 23 s"; then
    run prepare_tokenizer.py --out_dir "$DATA_DIR" --vocab_size 8000 --ja_docs 5000 --en_docs 5000
    wait_next
fi

# --- DEMO 2: Training data ---
if step 2 "Build the mixed Japanese/English binaries  ── about 2 s"; then
    run prepare_data.py --out_dir "$DATA_DIR" --tokenizer "$DATA_DIR/tokenizer.model" \
        --ja_tokens 3e6 --en_tokens 3e6 --val_ratio 0.02 --overwrite
    wait_next
fi

# --- DEMO 3: Training ---
# (the time estimate was measured on an RTX 4070 Ti SUPER; it will differ on a Mac)
if step 3 "Train the MoE model for $MAX_STEPS steps  ── about $(( (8000 + 68 * MAX_STEPS + 1999) / 2000 )) s"; then
    run macosx/train.py --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" --preset tiny --max_steps "$MAX_STEPS" \
        --batch_size 16 --grad_accum 2 --lr 2e-3 --warmup_steps 20 \
        --log_interval 25 --eval_interval 100 --eval_iters 5 \
        --sample_interval "$MAX_STEPS" --sample_tokens 24
    wait_next
fi

# --- DEMO 4: GGUF conversion ---
if step 4 "Convert to GGUF"; then
    run export_gguf.py --ckpt "$OUT_DIR/ckpt_best.pt" --tokenizer "$DATA_DIR/tokenizer.model" \
        --outtype f16 --output "$GGUF"
    wait_next
fi

# --- DEMO 5: Verification ---
if step 5 "Read the GGUF back and compare with the original model"; then
    run verify_gguf.py "$GGUF" --ckpt "$OUT_DIR/ckpt_best.pt" --tokenizer "$DATA_DIR/tokenizer.model" \
        --prompt "$PROMPT"
    wait_next
fi

# --- DEMO 6: llama.cpp ---
if step 6 "Generate with llama.cpp"; then
    if [ -n "$LLAMA" ]; then
        show llama-completion -m "$GGUF" -p "$PROMPT" -n 24 --temp 0 -no-cnv
        # Show only the lines from the startup log (stderr) confirming it was recognized as MoE; generated text goes to stdout
        LOG="$OUT_DIR/llama_cpp.log"
        "$LLAMA" -m "$GGUF" -p "$PROMPT" -n 24 --temp 0 -no-cnv -v 2> "$LOG"  # Without -v, print_info (model info) is not output
        echo ""
        # llama.cpp loads the model twice (for memory estimation), so drop duplicates
        grep -E "print_info: (arch|n_expert|n_expert_used|file type) " "$LOG" | sed 's/^.*print_info/print_info/' |
            awk '!seen[$0]++' | while IFS= read -r l; do printf "${GRAY}  %s${RESET_C}\n" "$l"; done
    else
        printf "${RED}llama-completion not found, skipping (set it with --llama-cpp-dir)${RESET_C}\n"
    fi
    wait_next
fi

# --- DEMO 7: Ollama ---
if step 7 "Register with Ollama and generate  ── first load takes about 20 s"; then
    if [ -n "$OLLAMA" ]; then
        # Written without a BOM, since Ollama can't read the FROM line if a BOM is present
        cat > "$OUT_DIR/Modelfile" <<EOF
FROM ./$(basename "$GGUF")
TEMPLATE "{{ .Prompt }}"
PARAMETER temperature 0
PARAMETER num_ctx 256
PARAMETER num_predict 24
EOF
        printf "${GRAY}--- %s/Modelfile ---${RESET_C}\n" "$OUT_DIR"
        sed 's/^/  /' "$OUT_DIR/Modelfile"
        (
            cd "$OUT_DIR" || exit 1
            show ollama create "$OLLAMA_MODEL" -f Modelfile
            "$OLLAMA" create "$OLLAMA_MODEL" -f Modelfile || { echo "Failed: ollama create" >&2; exit 1; }
            show ollama run "$OLLAMA_MODEL" "\"$PROMPT\""
            "$OLLAMA" run "$OLLAMA_MODEL" "$PROMPT"
        ) || exit 1
    else
        printf "${RED}ollama not found, skipping${RESET_C}\n"
    fi
fi

printf "\n${GREEN}All demo steps finished. Clean up: ollama rm %s${RESET_C}\n" "$OLLAMA_MODEL"
