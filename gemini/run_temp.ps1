# Set environment variables
$env:GOOGLE_API_KEY = ""
$env:PYTHONPATH = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

# Change to gemini directory
Set-Location $PSScriptRoot

# Resolve Python from PATH or use env override
$PythonExe = $env:PYTHON_CMD
if (-not $PythonExe) { $PythonExe = (Get-Command python -ErrorAction SilentlyContinue)?.Source }
if (-not $PythonExe) { $PythonExe = (Get-Command py -ErrorAction SilentlyContinue)?.Source }
if (-not $PythonExe) { $PythonExe = "python" }

Write-Host "Using Python: $PythonExe" -ForegroundColor Green
Write-Host "Running on ENTIRE datasets (no sample limit)" -ForegroundColor Green

# Run full evaluation on AMI dataset
Write-Host "--- Running full AMI dataset evaluation ---" -ForegroundColor Cyan
try {
    & $PythonExe run_eval.py --model_name="gemini/gemini-2.5-flash" --dataset_path="hf-audio/esb-datasets-test-only-sorted" --dataset="ami" --split="test"
    Write-Host "✓ AMI evaluation completed successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ AMI evaluation failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Run full multilingual evaluation
Write-Host "--- Running full multilingual evaluation (FLEURS English) ---" -ForegroundColor Cyan
try {
    & $PythonExe run_eval_ml.py --model_name="gemini/gemini-2.5-flash" --dataset="nithinraok/asr-leaderboard-datasets" --config_name="fleurs_en" --language="en" --split="test"
    Write-Host "✓ Multilingual evaluation completed successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ Multilingual evaluation failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "--- All full dataset evaluations completed successfully ---" -ForegroundColor Green
