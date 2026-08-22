# PowerShell version of run_gemini.sh (clean)

# Set environment variables
$env:PYTHONPATH = ".."

# Load .env if present to populate GOOGLE_API_KEY and others
try {
    $envFile = Join-Path $PSScriptRoot ".env"
    if (Test-Path $envFile) {
        Get-Content -Path $envFile | ForEach-Object {
            $line = $_.Trim()
            if ($line -and -not $line.StartsWith('#')) {
                if ($line -match '^(?<k>[^#=]+?)\s*=\s*(?<v>.*)$') {
                    $k = $Matches['k'].Trim()
                    $v = $Matches['v'].Trim()
                    if ($k) { Set-Item -Path Env:$k -Value $v | Out-Null }
                }
            }
        }
    }
} catch {}

# Check if Google API key is set
if (-not $env:GOOGLE_API_KEY) {
    Write-Host "Error: GOOGLE_API_KEY environment variable not set" -ForegroundColor Red
    Write-Host "Please set it with: `$env:GOOGLE_API_KEY='your_api_key_here'" -ForegroundColor Yellow
    exit 1
}

# Resolve Python from PATH (override by setting $env:PYTHON_CMD)
$PythonExe = $env:PYTHON_CMD
if (-not $PythonExe) { $PythonExe = (Get-Command python -ErrorAction SilentlyContinue)?.Source }
if (-not $PythonExe) { $PythonExe = (Get-Command py -ErrorAction SilentlyContinue)?.Source }
if (-not $PythonExe) { $PythonExe = "python" }
Write-Host "Using Python: $PythonExe" -ForegroundColor Green

$MODEL_IDs = @(
  "gemini/gemini-2.5-pro",
  "gemini/gemini-2.5-flash"
)

$BATCH_SIZE = 8
$MAX_SAMPLES = 2
Write-Host "Test samples per dataset: $MAX_SAMPLES" -ForegroundColor Green

foreach ($MODEL_ID in $MODEL_IDs) {
  Write-Host "--- Running Benchmarks for $MODEL_ID ---" -ForegroundColor Cyan

  # Quick setup test
  Write-Host "--- Testing setup with AMI dataset ---" -ForegroundColor Yellow
  try {
    & $PythonExe run_eval.py `
      --model_id="$MODEL_ID" `
      --dataset_path="hf-audio/esb-datasets-test-only-sorted" `
      --dataset="ami" `
      --split="test" `
      --max_eval_samples=1
    Write-Host "✓ Setup verified, continuing with benchmarks" -ForegroundColor Green
  } catch {
    Write-Host "✗ Setup test failed for $MODEL_ID, skipping..." -ForegroundColor Red
    continue
  }

  # English benchmarks
  Write-Host "--- Running English Benchmarks ---" -ForegroundColor Cyan
  $datasets = @("ami", "earnings22", "gigaspeech", "librispeech", "spgispeech", "tedlium", "voxpopuli")
  foreach ($dataset in $datasets) {
    Write-Host "Processing dataset: $dataset" -ForegroundColor Yellow
    if ($dataset -eq "librispeech") {
      foreach ($split in @("test.clean", "test.other")) {
        Write-Host "Running $dataset/$split..." -ForegroundColor White
        try {
          & $PythonExe run_eval.py `
            --model_id="$MODEL_ID" `
            --dataset_path="hf-audio/esb-datasets-test-only-sorted" `
            --dataset="$dataset" `
            --split="$split" `
            --batch_size=$BATCH_SIZE `
            --max_eval_samples=$MAX_SAMPLES
          Write-Host "✓ Completed $dataset/$split" -ForegroundColor Green
        } catch {
          Write-Host "✗ Warning: Failed to process $dataset/$split" -ForegroundColor Red
        }
      }
    } else {
      Write-Host "Running $dataset..." -ForegroundColor White
      try {
        & $PythonExe run_eval.py `
          --model_id="$MODEL_ID" `
          --dataset_path="hf-audio/esb-datasets-test-only-sorted" `
          --dataset="$dataset" `
          --split="test" `
          --batch_size=$BATCH_SIZE `
          --max_eval_samples=$MAX_SAMPLES
        Write-Host "✓ Completed $dataset" -ForegroundColor Green
      } catch {
        Write-Host "✗ Warning: Failed to process $dataset" -ForegroundColor Red
      }
    }
  }

  # Multilingual
  Write-Host "--- Running Multilingual Benchmarks ---" -ForegroundColor Cyan
  $EVAL_DATASETS = @{
    "fleurs" = @("en", "de", "fr", "it", "es", "pt")
    "mcv"    = @("en", "de", "es", "fr", "it")
    "mls"    = @("es", "fr", "it", "pt")
  }
  foreach ($ds in $EVAL_DATASETS.Keys) {
    foreach ($lang in $EVAL_DATASETS[$ds]) {
      $config = "${ds}_${lang}"
      Write-Host "Running evaluation for $config" -ForegroundColor White
      try {
        & $PythonExe run_eval_ml.py `
          --model_id="$MODEL_ID" `
          --dataset="nithinraok/asr-leaderboard-datasets" `
          --config_name="$config" `
          --language="$lang" `
          --split="test" `
          --max_eval_samples=$MAX_SAMPLES
      } catch {
        Write-Host "✗ Warning: Failed to process $config" -ForegroundColor Red
      }
    }
  }

  # Scoring
  Write-Host "--- Scoring results for $MODEL_ID ---" -ForegroundColor Cyan
  $RUNDIR = Get-Location
  Set-Location "..\normalizer"
  & $PythonExe -c "import eval_utils; eval_utils.score_results('$RUNDIR\results', '$MODEL_ID')"
  Set-Location $RUNDIR
}

Write-Host "--- All benchmarks complete ---"
