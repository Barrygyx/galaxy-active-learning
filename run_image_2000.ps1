param(
    [int[]]$Seeds = @(42),
    [int]$Threads = 4,
    [switch]$SkipInstall
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Extract all package contents into the original galaxy-active-learning project root." }
if (-not (Test-Path (Join-Path $PSScriptRoot "al_config.json"))) { throw "Original al_config.json is missing." }
if (-not $SkipInstall) {
    & $Python -c "import torch, torchvision"
    if ($LASTEXITCODE -ne 0) { throw "Existing torch/torchvision cannot import. Restore the working environment used for the 512-label demo." }
    & $Python -m pip install -r (Join-Path $PSScriptRoot "image_budget_runner\requirements_image_demo.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}
$RunArgs = @((Join-Path $PSScriptRoot "run_image_2000.py"), "--threads", "$Threads", "--seeds")
$RunArgs += @($Seeds | ForEach-Object { "$_" })
& $Python @RunArgs
if ($LASTEXITCODE -ne 0) { throw "Comparison stopped. Check the error and rerun the same command to resume." }
Write-Host "Finished. Open outputs\latest_image_2000.json for the report directory."
