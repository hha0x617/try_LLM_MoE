# try_LLM_MoE — Train a Homemade MoE LLM on Japanese/English Wikipedia and Export It as GGUF

**English** | [日本語](README_ja.md)

This project trains a Mixture of Experts (MoE) language model from scratch using the Japanese and English editions of Wikipedia.
The finished model, tokenizer included, is exported as **a single GGUF file**.

The GGUF is written with a Mixtral-compatible structure (`general.architecture = llama` + `llama.expert_count`),
so **it can be loaded and run directly in llama.cpp** (verified with llama.cpp b11756).

## Overview

```
download_wiki.py     Save Wikipedia Parquet files to data/raw/ (once; reused afterwards)
      ↓
prepare_tokenizer.py Train the SentencePiece vocabulary          -> data/tokenizer.model
      ↓
prepare_data.py      Build mixed Japanese/English token binaries -> data/train.bin, val*.bin, meta.json
      ↓
train.py             Train the MoE model                         -> out/ckpt.pt, out/ckpt_best.pt
      ↓
export_gguf.py       Checkpoint + vocabulary -> GGUF             -> out/wiki-moe-ja-en-f16.gguf
      ↓
verify_gguf.py       Inspect the GGUF and check it matches the original model (with --llama_cpp, also run inference in llama.cpp)
      ↓
eval_facts.py        (Optional) Measure answer accuracy with a fact-completion test -> out/eval_facts_baseline.json
```

Running `run_pipeline.ps1` executes the steps from `download_wiki.py` through `verify_gguf.py` in one go (the llama.cpp inference check and `eval_facts.py` are not included; run them separately if needed).

## Requirements

All results in this document were measured in the environment below.

| Item | Tested environment | Notes |
|---|---|---|
| OS | Windows 11 Pro (64-bit) | `run_pipeline.ps1` / `demo.ps1` are PowerShell scripts, and paths in the examples use Windows notation. Linux / macOS have not been tested |
| Shell | PowerShell | If blocked by the execution policy, use `-ExecutionPolicy Bypass` (see below) |
| Python | 3.14.0 (64-bit) | |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (16GB VRAM) | An NVIDIA GPU with CUDA is effectively required for training. 12GB works with a smaller batch (see "Training Time Estimates") |
| NVIDIA driver | 616.92 | Must support CUDA 12.8 |
| PyTorch | 2.11.0 + CUDA 12.8 (`cu128`) | Installed separately from `requirements.txt` |
| Other Python packages | numpy 2.5.4, datasets 5.1.0, huggingface_hub 2.2.0, pyarrow 26.0.0, sentencepiece 0.2.2, gguf 0.19.0, tqdm 4.70.1 | `requirements.txt` |
| Free disk space | 20GB or more recommended | Measured for a full run: Wikipedia Parquet ~6.0GB, `data/*.bin` ~2.0GB, `out/` ~4.6GB (two checkpoints of ~2.1GB each + 356MB GGUF) |
| Network | Required for the first download | About 5.4GB from Hugging Face. Not needed afterwards |

Optional tools for using the GGUF:

| Tool | Tested version |
|---|---|
| llama.cpp | b11756 (`win-cpu-x64`; `win-cuda` builds also available) |
| Ollama | 0.40.6 |

To run on macOS (Apple Silicon), see "[Running on macOS](#running-on-macos)".

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Use Python 3.14 as `python`. All scripts are expected to run with the Python in `.venv` (`run_pipeline.ps1` and `demo.ps1` call `.venv\Scripts\python.exe` directly).

## Usage

### Run everything at once

```powershell
.\run_pipeline.ps1 -Smoke            # Smoke test. Finishes in a few minutes (output: smoke\)
.\run_pipeline.ps1                   # Full run. small preset, 20000 steps
.\run_pipeline.ps1 -SkipTo train     # Resume from the specified step
```

If the script is blocked by the execution policy, run it with `powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1`.

### Run step by step

```powershell
$py = ".\.venv\Scripts\python"
& $py download_wiki.py --en_shards 8            # Japanese: all 15 files (3.9GB); English: 8 of 41 files (~2.3GB)
& $py prepare_tokenizer.py                      # Vocabulary of 32000. Takes 10–20 minutes
& $py prepare_data.py --ja_tokens 5e8 --en_tokens 5e8
& $py train.py --preset small --max_steps 20000
& $py generate.py --prompt "日本の首都は" --show_routing
& $py export_gguf.py --ckpt out\ckpt_best.pt --outtype f16
& $py verify_gguf.py out\wiki-moe-ja-en-f16.gguf --ckpt out\ckpt_best.pt
& $py verify_gguf.py out\wiki-moe-ja-en-f16.gguf --llama_cpp D:\tools\llama.cpp   # (Optional) also run inference in llama.cpp and compare
& $py eval_facts.py --ckpt out\ckpt_best.pt --ckpt out\ckpt.pt --json out\eval_facts_baseline.json   # (Optional) answer accuracy
```

For `--llama_cpp`, specify the folder where the llama.cpp release (`llama-completion.exe`, etc.) is extracted.

(`日本の首都は` = "The capital of Japan is")

## Running on macOS

On Apple Silicon Macs, use the Mac scripts in `macosx/`. Training uses the Apple GPU (MPS).

> **Note:** The Mac scripts have only been checked on Windows using the CPU; they have not been tested on an actual Mac.

### Target environment

| Item | Target |
|---|---|
| Mac | M2 Pro (32GB memory) or M4 (32GB memory) |
| macOS | 14 Sonoma or later (for bf16 on MPS) |
| Python | 3.14 (python.org or Homebrew) |
| PyTorch | 2.11 (the standard build, with MPS support) |

### Setup

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install torch
.venv/bin/python -m pip install -r requirements.txt
chmod +x macosx/*.sh    # if the scripts are not executable
```

### Usage

Run the commands from the repository root. Options mean the same as in the Windows version, written in lowercase with hyphens.

```bash
./macosx/run_pipeline.sh --smoke             # Smoke test (output: smoke/)
./macosx/run_pipeline.sh                     # Full run. small preset, 20000 steps
./macosx/run_pipeline.sh --skip-to train     # Resume from the specified step
./macosx/run_pipeline.sh --batch-size 4 --grad-accum 16   # If memory runs short (same effective batch of 64)
./macosx/demo.sh                             # Live demo for presentations (same 7 steps as demo.ps1)
```

To run step by step, use the `macosx/` versions only for training, generation, and the answer accuracy test. The other scripts are shared with Windows.

```bash
py=.venv/bin/python
$py download_wiki.py --en_shards 8
$py prepare_tokenizer.py
$py prepare_data.py --ja_tokens 5e8 --en_tokens 5e8
$py macosx/train.py --preset small --max_steps 20000
$py macosx/generate.py --prompt "日本の首都は" --show_routing
$py export_gguf.py --ckpt out/ckpt_best.pt --outtype f16
$py verify_gguf.py out/wiki-moe-ja-en-f16.gguf --ckpt out/ckpt_best.pt
$py macosx/eval_facts.py --ckpt out/ckpt_best.pt --json out/eval_facts_baseline.json
```

### Differences from the Windows version

| File | Change |
|---|---|
| `macosx/train.py` | Uses the Apple GPU (MPS), or the CPU if unavailable (then switches to `--dtype fp32`). Creates the GradScaler for `--dtype fp16` for MPS |
| `macosx/generate.py`, `macosx/eval_facts.py` | Use the Apple GPU (MPS), or the CPU if unavailable |
| `macosx/moe_model.py` | Computes RoPE with real-valued cos/sin instead of complex numbers (complex ops are not fully supported on MPS). The results are identical |
| `macosx/run_pipeline.sh`, `macosx/demo.sh` | bash versions of `run_pipeline.ps1` / `demo.ps1` (written without bash 4+ features so they run on macOS's built-in bash 3.2). Training uses `macosx/train.py` |

- Checkpoints and GGUF files use the same format as the Windows version and can be loaded by either. For a checkpoint trained on Windows, the output difference with `macosx/moe_model.py` is at most about 0.00002 (checked with `out/ckpt_best.pt`).
- To use a Python other than `.venv`, set its path in the `PYTHON` environment variable (e.g. `PYTHON=python3.14 ./macosx/run_pipeline.sh --smoke`).
- For llama.cpp, use e.g. the `macos-arm64` build from the [releases page](https://github.com/ggml-org/llama.cpp/releases). In `demo.sh`, specify its location with `--llama-cpp-dir` (searched on PATH if omitted).

### Memory and training time estimates

- The small preset's default (`--batch_size 8`) used about 10GB of VRAM on the RTX 4070 Ti SUPER, so it should fit on a Mac with 32GB of memory. If memory runs short, reduce it while keeping the effective batch, e.g. `--batch-size 4 --grad-accum 16`.
- Training speed on a Mac has not been measured. Expect it to be much slower than the RTX 4070 Ti SUPER; first run `--smoke` to check throughput (`tok/s`) and remaining time (`ETA`), and reduce `--max-steps` if needed.

## Reusing Downloaded Data

Each stage leaves its output on disk, so later runs do not need to redo earlier stages.

| Stage | Location | How it is reused |
|---|---|---|
| Raw Wikipedia | `data/raw/wikipedia/20231101.{ja,en}/*.parquet` | If these files exist, every script reads them locally. Otherwise data is streamed from Hugging Face (which means downloading every time) |
| Tokenizer | `data/tokenizer.model` | `run_pipeline.ps1` skips this step if it already exists |
| Token sequences | `data/train.bin`, etc. | `prepare_data.py` does not overwrite existing files (use `--overwrite` to rebuild) |
| Training progress | `out/ckpt.pt` | `train.py --resume` continues training. A checkpoint is also saved when stopped with Ctrl+C |

- Re-running `download_wiki.py` skips files already downloaded. An interrupted download resumes where it left off.
- To add more English shards later, re-run with e.g. `--en_shards 16`.
- If only some shards exist locally, only those shards are used. If the data runs out before reaching the target token count, `prepare_data.py` appends `(data exhausted)` to that language's summary line.

## Model Architecture (`moe_model.py`)

| Preset | Layers | d_model | Experts (top-k) | d_ff per expert | Total params | Active params |
|---|---|---|---|---|---|---|
| tiny  | 2  | 128 | 4 (2) | 256  | 9M   | 9M   |
| small | 8  | 512 | 8 (2) | 1408 | 178M | 74M  |
| base  | 12 | 768 | 8 (2) | 2048 | 521M | 181M |

(Parameter counts assume a vocabulary of 32000.)

- Each layer is RMSNorm → GQA Attention (RoPE) → RMSNorm → MoE FFN.
- The MoE FFN uses a softmax router to select the top 2 experts, renormalizes their weights, and passes the input to SwiGLU experts (the same scheme as Mixtral).
- During training, a load-balancing loss (Switch Transformer style) is added to keep expert usage from becoming skewed.
- Expert weights are held together in a 3D tensor of shape `[n_expert, d_ff, d_model]` and written directly to GGUF as `blk.N.ffn_{gate,up,down}_exps.weight`.

## Training Time Estimates (RTX 4070 Ti SUPER 16GB)

- With the small preset (batch 8 × accumulation 8 × 1024 tokens ≈ 65k tokens per step), measured throughput is about **51k tokens/s** with about **10GB** of VRAM.
- 20000 steps (~1.3B tokens) take about **8 hours**. Measured times for a default `run_pipeline.ps1` run:

  | Step | Time |
  |---|---|
  | [0] Download (5.4GB of Parquet) | ~4 min (depends on connection; 0 min on later runs) |
  | [1] Vocabulary (32,000 tokens) | ~1.5 min |
  | [2] Training data (1B tokens) | ~2.5 min |
  | [3] Training (20000 steps) | 7 h 58 min |
  | [4] GGUF conversion and verification | ~40 s |
- `--batch_size 16` exceeds 16GB of VRAM and spills into shared memory, slowing training to about 1/4. To increase the effective batch, raise `--grad_accum` instead.
- For an RTX 3060 12GB, use `--batch_size 4 --grad_accum 16` as a starting point.

## What You Can Monitor During Training

- **Every 20 steps**: loss, load-balancing loss, learning rate, throughput, remaining time, expert utilization (averaged over all layers)
- **Every 500 steps**: validation loss (mixed, Japanese only, English only) and per-layer expert utilization (shown separately for Japanese and English)
- **Every 1000 steps**: generated samples from prompts
- Logs are also appended to `out/train_log.jsonl`.

Even in the smoke test (tiny, 300 steps), layer 1 already showed language specialization: "Japanese uses Experts 0 and 2, English uses Experts 1 and 3."

```
L1  ja [45.2  1.2 49.1  4.5] | en [ 9.1 43.3  3.5 44.1]
```

## Using the GGUF

```powershell
# llama.cpp (win-cuda / win-cpu builds from https://github.com/ggml-org/llama.cpp/releases)
llama-completion -m out\wiki-moe-ja-en-f16.gguf -p "日本の首都は" -n 64 -no-cnv

# To quantize further
llama-quantize out\wiki-moe-ja-en-f16.gguf out\wiki-moe-ja-en-q4_k_m.gguf Q4_K_M
```

- `export_gguf.py --outtype` accepts `f32`, `f16`, `bf16`, and `q8_0`.
- RMSNorm and router weights are always written in F32 to preserve precision.
- The small preset's d_ff = 1408 is not a multiple of 256, so with K-quants such as Q4_K, llama-quantize switches to a different quantization type for `ffn_down_exps` only.
- This is a pretrained-only base model; it has not been instruction-tuned (SFT). Use it to continue text, not as a chat model.

### Using with Ollama

Create `out\Modelfile` with the following content and register it with `ollama create`. The full small model was trained with a 1024-token context, so set `num_ctx` to 1024 (256 for the tiny demo model).

```
FROM ./wiki-moe-ja-en-f16.gguf
TEMPLATE "{{ .Prompt }}"
PARAMETER temperature 0
PARAMETER num_ctx 1024
PARAMETER num_predict 40
```

```powershell
cd out
ollama create wiki-moe -f Modelfile
ollama run wiki-moe "日本の首都は"
ollama rm wiki-moe    # Clean up
```

- Save the Modelfile as UTF-8 without BOM.
- Because this is a base model, TEMPLATE passes the prompt through unchanged.

### Results for the Full Model

The model trained for 20000 steps (the checkpoint at step 19500, which had the best validation loss) was converted to GGUF (f16, 356MB) and tested. Generation uses temperature 0 (greedy).

| Input | Continuation (beginning) |
|---|---|
| 日本の首都は ("The capital of Japan is") | 、北海道札幌市東区にある首都。概要 首都は、札幌市東区にある首都の1つである。… (", a capital located in Higashi-ku, Sapporo, Hokkaido. Overview: The capital is one of the capitals located in Higashi-ku, Sapporo. …") |
| 富士山は ("Mount Fuji is") | 、長野県長野市富士見町富士見にある富士山。標高は1,000m。概要 … (", Mount Fuji located in Fujimi, Fujimi-machi, Nagano City, Nagano Prefecture. Elevation 1,000 m. Overview …") |
| The capital of France is | the capital of the French colony of Saint-Denis. History … |
| Albert Einstein was | a German-born American film director, screenwriter, and film producer. Life and career … |

- **Consistency:** For all four inputs, the outputs of PyTorch, llama.cpp (b11756 win-cpu-x64), and Ollama 0.40.6 were identical.
- **Conversion error:** `verify_gguf.py` compared a model rebuilt from the GGUF against the checkpoint. The maximum logits difference was 0.008, and next-token predictions agreed 100%.
- **Output quality:** The model has picked up Wikipedia-like style, such as "Overview" headings and biography phrasing. However, the facts are wrong (a limitation of 180M parameters and 1.3B tokens).

### Answer Accuracy Test

```powershell
.venv\Scripts\python.exe eval_facts.py --ckpt out\ckpt_best.pt --ckpt out\ckpt.pt --verbose --json out\eval_facts_baseline.json
```

With `--json`, the results (summary and per-question results) are saved to `out\eval_facts_baseline.json`; without it they are only printed. If `--ckpt` is omitted, only `out\ckpt_best.pt` is evaluated.

For the 100 questions in `eval_facts.jsonl` (e.g. "日本の首都は → 東京", "The capital of France is → Paris"; 50 each in Japanese and English), the script computes from the logits the probability that the correct answer immediately follows the prompt. No sampling is involved, so results are deterministic.

| Metric | Meaning |
|---|---|
| Accuracy | Fraction where greedy decoding (temperature 0) outputs the correct answer exactly |
| Top-10 | Fraction where the first token of the answer ranks in the top 10 out of 32,000 |
| Median rank | Median rank of the first token of the answer |
| Answer prob. | Geometric mean of the probability assigned to the full answer |

Results for the full model in the author's environment (see "Requirements") are shown below. Seeds are fixed, but GPU computation is not fully deterministic, so a model trained in another environment may give slightly different values.

| Checkpoint | Language | Accuracy | Top-10 | Median rank | Answer prob. |
|---|---|---|---|---|---|
| ckpt_best.pt (19500) | Japanese | 4% | 48% | 11 | 0.0023 |
| | English | 2% | 36% | 16 | 0.0010 |
| | Overall | 3% | 42% | 14 | 0.0015 |
| ckpt.pt (20000) | Overall | 2% | 37% | 15 | 0.0012 |

## Main Changes from the Original Plan

| Original plan | This implementation | Reason |
|---|---|---|
| `cl-tohoku/wikipedia-ja` and the legacy `wikipedia` script | `wikimedia/wikipedia` (20231101.ja / .en) | The same official dataset for both languages, and the Parquet files can be saved and reused as-is |
| Borrow the vocabulary from Gemma-2 (256k vocab) | Train our own vocabulary (`tokenizer.model`, 32k vocab). Both use SentencePiece as the tokenizer | Gemma-2 is a gated model and requires an access request (*). Also, with a 256k vocab in the small configuration, the token tables (embedding + output layer) would be 262M parameters, 64% of the model, and the output-layer activations alone would use 8.4GB of VRAM per batch (vs. 33M, 18%, and 1.0GB with a 32k vocab) |
| uint32 token sequences | uint16 token sequences | The vocabulary is smaller than 65536, which halves the file size |
| A single layer: Embedding → MoE → LM Head | Multi-layer transformer with attention | Without attention the model cannot use context and does not work as a language model |
| GGUF architecture `custom-moe` | `llama` + `expert_count` (Mixtral-compatible) | So that existing inference engines (llama.cpp) can run it as-is |
| Hyperparameters hard-coded in the conversion script | Model config bundled into the checkpoint | Prevents mismatches between the training-time and export-time model shape |

\* Gemma-2 is published on Hugging Face as a "gated model", so you must request access (accept the license). You cannot download it without accepting the license on the model page and logging in with an access token.

### Borrowing the Gemma-2 Vocabulary as Originally Planned (`borrow_tokenizer.py`)

Only one file is borrowed: `tokenizer.model` (the SentencePiece vocabulary, about 4MB). No model weights are used, so this is not distillation.

1. Open https://huggingface.co/google/gemma-2-2b and accept the license (Gemma Terms of Use).
2. Create an access token with Read permission at https://huggingface.co/settings/tokens and log in:
   ```powershell
   .venv\Scripts\hf.exe auth login
   ```
3. Fetch the vocabulary and continue through training (outputs go to `data_gemma\` and `out_gemma\`; your own `data\` and `out\` are not overwritten):
   ```powershell
   .\run_pipeline.ps1 -GemmaVocab               # Reuses already-downloaded Parquet files
   # Or step by step
   python borrow_tokenizer.py --out_dir data_gemma
   python prepare_data.py --out_dir data_gemma --tokenizer data_gemma\tokenizer.model
   python train.py --data_dir data_gemma --out_dir out_gemma --batch_size 2 --grad_accum 32
   ```

The scripts automatically handle the differences from the self-trained vocabulary.

| Item | Self-trained (32k) | Gemma-2 (256k) | Handling |
|---|---|---|---|
| Token dtype | uint16 | uint32 (`train.bin` about 2× larger) | `prepare_data.py` switches based on vocab size; `train.py` reads `dtype` from `meta.json` |
| Leading space prefix | Yes | No | `export_gguf.py` detects it and writes `add_space_prefix` |
| Special token IDs | unk 0 / bos 1 / eos 2 | pad 0 / eos 1 / bos 2 / unk 3 | All scripts look them up via `sp.bos_id()` etc. |
| VRAM (small) | `--batch_size 8` | Output layer is 8× larger, so `--batch_size 2 --grad_accum 32` (same effective batch of 64) | Set automatically by `run_pipeline.ps1 -GemmaVocab` |

Notes:
- Total parameters grow to about 407M (262M of which are token tables), so both training time and GGUF size increase (not measured).
- The vocabulary is part of Gemma's distribution. If you publish a model or GGUF built with it, follow the conditions of the Gemma Terms of Use (such as passing on its use restrictions).
- You can also pass an already-downloaded `tokenizer.model` with `--file`.

## Files

### Runners

| File | Purpose |
|---|---|
| `run_pipeline.ps1` | Runs the whole pipeline |
| `demo.ps1` | Live demo for presentations. Runs 7 steps one at a time on Enter (`-Reset` clears previous output, `-NoPause` for rehearsal). Requires the Parquet files to be downloaded first with `download_wiki.py` |

### Pipeline (in order)

| File | Purpose |
|---|---|
| `download_wiki.py` | Downloads the Parquet files |
| `prepare_tokenizer.py` | Trains SentencePiece |
| `borrow_tokenizer.py` | Alternative to `prepare_tokenizer.py`: fetches the Gemma-2 vocabulary from Hugging Face (requires license acceptance and login) |
| `prepare_data.py` | Builds the mixed Japanese/English binaries |
| `train.py` | Training (with monitoring output, checkpointing, and resume) |
| `export_gguf.py` | Conversion to GGUF |

### Shared modules

| File | Purpose |
|---|---|
| `moe_model.py` | The MoE transformer (used by `train.py`, `export_gguf.py`, `generate.py`, `verify_gguf.py`, `eval_facts.py`) |
| `wiki_common.py` | Reads Wikipedia articles (streams them if local Parquet files are missing). Used by `download_wiki.py`, `prepare_tokenizer.py`, `prepare_data.py` |

### Evaluation and verification

| File | Purpose |
|---|---|
| `verify_gguf.py` | GGUF verification |
| `generate.py` | Text generation and routing visualization |
| `eval_facts.py` / `eval_facts.jsonl` | Measures answer accuracy with a fact-completion test (100 questions in Japanese and English). Pass multiple `--ckpt` options to compare side by side; `--json` saves the results |

### macOS (`macosx/`)

| File | Purpose |
|---|---|
| `macosx/run_pipeline.sh` / `macosx/demo.sh` | bash versions of `run_pipeline.ps1` / `demo.ps1` |
| `macosx/train.py` / `macosx/generate.py` / `macosx/eval_facts.py` | Versions that use the Apple GPU (MPS) |
| `macosx/moe_model.py` | Version that computes RoPE with real-valued math (used by the three above) |

### Other

| File | Purpose |
|---|---|
| `requirements.txt` | Dependencies other than PyTorch |
| `.gitattributes` | Keeps `.sh` files with LF line endings |
| `LICENSE` | MIT License |

## License

The code in this repository is released under the [MIT License](LICENSE).

The following are not covered by this license; follow their respective terms:

- Wikipedia text used as training data: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) / GFDL
- The Gemma-2 vocabulary obtained with `borrow_tokenizer.py`: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
