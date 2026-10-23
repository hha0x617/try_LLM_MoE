# Live demo script for the presentation (runs the 7 steps of the "live demo" slides as-is)
#
#   .\demo.ps1                          # Advance one step at a time with Enter
#   .\demo.ps1 -Reset                   # Delete the previous demo folder and Ollama model before starting
#   .\demo.ps1 -NoPause                 # Run straight through without pausing (for rehearsal)
#   .\demo.ps1 -StartAt 4               # Resume from an intermediate step
#   .\demo.ps1 -StartAt 3 -MaxSteps 400 # Reuse the built data and try a different number of training steps
#   .\demo.ps1 -LlamaCppDir D:\tools\llama.cpp   # Location of llama-completion.exe (searched on PATH if omitted)
#
# Prerequisite: Parquet files already downloaded to data\raw\wikipedia\ (download_wiki.py)
param(
    [switch]$Reset,
    [switch]$NoPause,
    [ValidateRange(1, 7)]
    [int]$StartAt = 1,
    [ValidateRange(10, 100000)]
    [int]$MaxSteps = 200,          # Number of training steps for DEMO 3
    [string]$LlamaCppDir = "",
    [string]$Prompt = "日本の首都は",
    [string]$OllamaModel = "demo-moe"
)
# Check external command success via $LASTEXITCODE (not Stop, since Windows PowerShell 5.1 treats stderr output as errors)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$DataDir = "demo\data"
$OutDir = "demo\out"
$Gguf = "$OutDir\demo-moe-f16.gguf"

function Step([int]$n, [string]$title) {
    if ($n -lt $StartAt) { return $false }
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkGray
    Write-Host " DEMO $n / 7 : $title" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkGray
    return $true
}

function Show([string[]]$cmd) {
    Write-Host ("PS> " + ($cmd -join " ")) -ForegroundColor Yellow
}

function Run([string[]]$argsList) {
    Show (@("python") + $argsList)
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $py @argsList
    if ($LASTEXITCODE -ne 0) { throw "Failed: python $($argsList -join ' ')" }
    Write-Host ("  ({0:N1} s)" -f $sw.Elapsed.TotalSeconds) -ForegroundColor DarkGray
}

function Wait-Next {
    if (-not $NoPause) { [void](Read-Host "`n[Enter] to continue") }
}

# --- Preflight checks ---
if (-not (Test-Path $py)) { throw ".venv not found: $py" }
if (-not (Get-ChildItem "data\raw\wikipedia\*\*.parquet" -ErrorAction SilentlyContinue)) {
    throw "No Parquet files in data\raw\wikipedia. Run download_wiki.py --ja_shards 1 --en_shards 1 first"
}
$llama = $null
if ($LlamaCppDir) {
    $llama = Get-ChildItem -Path $LlamaCppDir -Recurse -Filter "llama-completion.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
} else {
    $llama = (Get-Command llama-completion -ErrorAction SilentlyContinue).Source
}
$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
Write-Host "llama.cpp : $(if ($llama) { $llama } else { 'not found (skipping DEMO 6)' })"
Write-Host "Ollama    : $(if ($ollama) { $ollama } else { 'not found (skipping DEMO 7)' })"

if ($Reset) {
    if (Test-Path "demo") { Remove-Item -Recurse -Force "demo"; Write-Host "Deleted the demo folder" }
    if ($ollama) { & $ollama rm $OllamaModel 2>$null | Out-Null; Write-Host "Removed Ollama model $OllamaModel (if registered)" }
}
New-Item -ItemType Directory -Force $DataDir, $OutDir | Out-Null

# --- DEMO 1: Vocabulary ---
if (Step 1 "Train the vocabulary (SentencePiece)  ── about 23 s") {
    Run @("prepare_tokenizer.py", "--out_dir", $DataDir, "--vocab_size", "8000", "--ja_docs", "5000", "--en_docs", "5000")
    Wait-Next
}

# --- DEMO 2: Training data ---
if (Step 2 "Build the mixed Japanese/English binaries  ── about 2 s") {
    Run @("prepare_data.py", "--out_dir", $DataDir, "--tokenizer", "$DataDir\tokenizer.model",
          "--ja_tokens", "3e6", "--en_tokens", "3e6", "--val_ratio", "0.02", "--overwrite")
    Wait-Next
}

# --- DEMO 3: Training ---
if (Step 3 ("Train the MoE model for {0} steps  ── about {1} s" -f $MaxSteps, [math]::Ceiling(4 + 6.8 * $MaxSteps / 200))) {
    Run @("train.py", "--data_dir", $DataDir, "--out_dir", $OutDir, "--preset", "tiny", "--max_steps", "$MaxSteps",
          "--batch_size", "16", "--grad_accum", "2", "--lr", "2e-3", "--warmup_steps", "20",
          "--log_interval", "25", "--eval_interval", "100", "--eval_iters", "5",
          "--sample_interval", "$MaxSteps", "--sample_tokens", "24")
    Wait-Next
}

# --- DEMO 4: GGUF conversion ---
if (Step 4 "Convert to GGUF") {
    Run @("export_gguf.py", "--ckpt", "$OutDir\ckpt_best.pt", "--tokenizer", "$DataDir\tokenizer.model",
          "--outtype", "f16", "--output", $Gguf)
    Wait-Next
}

# --- DEMO 5: Verification ---
if (Step 5 "Read the GGUF back and compare with the original model") {
    Run @("verify_gguf.py", $Gguf, "--ckpt", "$OutDir\ckpt_best.pt", "--tokenizer", "$DataDir\tokenizer.model",
          "--prompt", $Prompt)
    Wait-Next
}

# --- DEMO 6: llama.cpp ---
if (Step 6 "Generate with llama.cpp") {
    if ($llama) {
        $cmd = @("-m", $Gguf, "-p", $Prompt, "-n", "24", "--temp", "0", "-no-cnv")
        Show (@("llama-completion") + $cmd)
        # Show only the lines from the startup log (stderr) confirming it was recognized as MoE; generated text goes to stdout
        $log = Join-Path $OutDir "llama_cpp.log"
        & $llama @cmd -v 2> $log  # Without -v, print_info (model info) is not output
        Write-Host ""
        Select-String -Path $log -Pattern "print_info: (arch|n_expert|n_expert_used|file type) " |
            ForEach-Object { $_.Line -replace '^.*?print_info', 'print_info' } |
            Select-Object -Unique |  # llama.cpp loads the model twice (for memory estimation), so drop duplicates
            ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
    } else {
        Write-Host "llama-completion not found, skipping (set it with -LlamaCppDir)" -ForegroundColor Red
    }
    Wait-Next
}

# --- DEMO 7: Ollama ---
if (Step 7 "Register with Ollama and generate  ── first load takes about 20 s") {
    if ($ollama) {
        $modelfile = @(
            "FROM ./$(Split-Path $Gguf -Leaf)",
            'TEMPLATE "{{ .Prompt }}"',
            "PARAMETER temperature 0",
            "PARAMETER num_ctx 256",
            "PARAMETER num_predict 24"
        )
        # Write as UTF-8 without BOM, since Ollama can't read the FROM line if a BOM is present
        [IO.File]::WriteAllLines((Join-Path $PSScriptRoot "$OutDir\Modelfile"), $modelfile, (New-Object Text.UTF8Encoding $false))
        Write-Host "--- $OutDir\Modelfile ---" -ForegroundColor DarkGray
        $modelfile | ForEach-Object { Write-Host "  $_" }
        Push-Location $OutDir
        try {
            Show @("ollama", "create", $OllamaModel, "-f", "Modelfile")
            & $ollama create $OllamaModel -f Modelfile
            if ($LASTEXITCODE -ne 0) { throw "Failed: ollama create" }
            Show @("ollama", "run", $OllamaModel, "`"$Prompt`"")
            & $ollama run $OllamaModel $Prompt
        } finally {
            Pop-Location
        }
    } else {
        Write-Host "ollama not found, skipping" -ForegroundColor Red
    }
}

Write-Host "`nAll demo steps finished. Clean up: ollama rm $OllamaModel" -ForegroundColor Green
