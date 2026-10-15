param(
    [ValidateSet("quick", "all")][string]$Suite = "quick",
    [int]$TrainSize = 1000,
    [int]$ImageSize = 128,
    [int]$Epochs = 0,
    [switch]$SkipInstall,
    [switch]$SmokeOnly
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Existing .venv missing. Extract this package into galaxy-active-learning alongside run_pipeline.py, data and outputs."
}
if (-not $SkipInstall) {
    & $Python -c "import torch, torchvision"
    if ($LASTEXITCODE -ne 0) {
        & $Python -m pip install "torch>=2.2,<3" "torchvision>=0.17,<0.30" --index-url https://download.pytorch.org/whl/cpu
        if ($LASTEXITCODE -ne 0) { throw "PyTorch/torchvision installation failed." }
    }
    & $Python -m pip install -r (Join-Path $PSScriptRoot "requirements_image_demo.txt")
    if ($LASTEXITCODE -ne 0) { throw "Image demo dependencies failed." }
}
$Arguments = @((Join-Path $PSScriptRoot "run_image_demo.py"), "--suite", $Suite,
               "--train-size", "$TrainSize", "--image-size", "$ImageSize")
if ($Epochs -gt 0) {
    $Minimum = [Math]::Min(6, $Epochs)
    $Warmup = [Math]::Min(2, [Math]::Max(0, $Epochs - 1))
    $Arguments += @("--epochs", "$Epochs", "--min-epochs", "$Minimum", "--warmup-epochs", "$Warmup")
}
if ($SmokeOnly) { $Arguments += "--smoke" }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Image demo stopped. Fix the error and rerun the same command to resume." }
& $Python (Join-Path $PSScriptRoot "analyze_image_ensemble.py")
if ($LASTEXITCODE -ne 0) { throw "CNN training completed; ensemble analysis failed. See error above." }
Write-Host "Complete. Result location: outputs\latest_image_demo.json"
