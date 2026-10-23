# Japanese/English Wikipedia MoE model pipeline (download -> tokenizer -> data -> train -> GGUF)
#
#   .\run_pipeline.ps1                     # Full settings (small preset / about 7-8 hours)
#   .\run_pipeline.ps1 -Smoke              # Quick sanity check that finishes in a few minutes
#   .\run_pipeline.ps1 -SkipTo train       # Resume from an intermediate step (download|tokenizer|data|train|export)
#   .\run_pipeline.ps1 -GemmaVocab         # Borrow the vocabulary from Gemma-2 (output to data_gemma\ / out_gemma\)
#
# Files already built (Parquet, tokenizer.model, train.bin) are reused automatically, not rebuilt.
param(
    [switch]$Smoke,
    [switch]$GemmaVocab,         # Requires accepting the license on Hugging Face and `hf auth login` beforehand
    [ValidateSet("download", "tokenizer", "data", "train", "export")]
    [string]$SkipTo = "download",
    [string]$Preset = "small",
    [int]$MaxSteps = 20000,
    [int]$EnShards = 8,          # Number of English shards to fetch (1 shard ≈ 280MB / ~100M tokens)
    [double]$JaTokens = 5e8,
    [double]$EnTokens = 5e8,
    [string]$OutType = "f16"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"   # Flush output incrementally even when saving to a file (e.g. via Tee-Object)

if ($Smoke) {
    $DataDir = "smoke\data"; $OutDir = "smoke\out"
    $Preset = "tiny"; $MaxSteps = 300; $EnShards = 1
    $JaTokens = 3e6; $EnTokens = 3e6
    $tokArgs = @("--vocab_size", "8000", "--ja_docs", "5000", "--en_docs", "5000")
    $trainArgs = @("--batch_size", "16", "--grad_accum", "2", "--lr", "2e-3", "--warmup_steps", "30",
                   "--log_interval", "50", "--eval_interval", "150", "--sample_interval", "300")
} else {
    $DataDir = "data"; $OutDir = "out"
    $tokArgs = @()
    $trainArgs = @()
}
if ($GemmaVocab) {
    # Output to separate folders so files don't mix with those built from our own vocabulary
    $DataDir = "${DataDir}_gemma"; $OutDir = "${OutDir}_gemma"
    # The 256K vocabulary makes the output layer 8x larger, so lower the micro-batch but keep the effective batch (64)
    if (-not $Smoke) { $trainArgs = @("--batch_size", "2", "--grad_accum", "32") }
}
$steps = @("download", "tokenizer", "data", "train", "export")
$start = $steps.IndexOf($SkipTo)

function Run($argsList) {
    & $py @argsList
    if ($LASTEXITCODE -ne 0) { throw "Failed: python $($argsList -join ' ')" }
}

if ($start -le 0) {
    Write-Host "`n=== [0] Download Wikipedia ===" -ForegroundColor Cyan
    $ja = if ($Smoke) { 1 } else { 0 }
    Run @("download_wiki.py", "--ja_shards", $ja, "--en_shards", $EnShards)
}
if ($start -le 1) {
    Write-Host "`n=== [1] Train the vocabulary ===" -ForegroundColor Cyan
    if (Test-Path "$DataDir\tokenizer.model") { Write-Host "Already built, reusing: $DataDir\tokenizer.model" }
    elseif ($GemmaVocab) { Run @("borrow_tokenizer.py", "--out_dir", $DataDir) }
    else { Run (@("prepare_tokenizer.py", "--out_dir", $DataDir) + $tokArgs) }
}
if ($start -le 2) {
    Write-Host "`n=== [2] Build the mixed Japanese/English binaries ===" -ForegroundColor Cyan
    if (Test-Path "$DataDir\train.bin") { Write-Host "Already built, reusing: $DataDir\train.bin" }
    else { Run @("prepare_data.py", "--out_dir", $DataDir, "--tokenizer", "$DataDir\tokenizer.model",
                 "--ja_tokens", $JaTokens, "--en_tokens", $EnTokens) }
}
if ($start -le 3) {
    Write-Host "`n=== [3] Train the MoE model ===" -ForegroundColor Cyan
    $resume = if (Test-Path "$OutDir\ckpt.pt") { @("--resume") } else { @() }
    Run (@("train.py", "--data_dir", $DataDir, "--out_dir", $OutDir, "--preset", $Preset,
           "--max_steps", $MaxSteps) + $trainArgs + $resume)
}
if ($start -le 4) {
    Write-Host "`n=== [4] Convert to GGUF and verify ===" -ForegroundColor Cyan
    $gguf = "$OutDir\wiki-moe-ja-en-$OutType.gguf"
    Run @("export_gguf.py", "--ckpt", "$OutDir\ckpt_best.pt", "--tokenizer", "$DataDir\tokenizer.model",
          "--outtype", $OutType, "--output", $gguf)
    Run @("verify_gguf.py", $gguf, "--ckpt", "$OutDir\ckpt_best.pt", "--tokenizer", "$DataDir\tokenizer.model")
    Write-Host "`nDone: $gguf" -ForegroundColor Green
}
