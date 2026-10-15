param([switch]$SkipInstall, [switch]$SmokeOnly)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$PythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $PythonExe)) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv. Check Python installation.' }
}
if (-not $SkipInstall) {
    # Install a compatible CPU pair together; no NVIDIA GPU or CUDA is required.
    & $PythonExe -m pip install 'torch>=2.2,<3' 'torchvision>=0.17,<0.30' --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed. Check internet access and rerun.' }
    & $PythonExe -m pip install -r requirements_active_learning.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
Write-Host 'Stage 1/2: small smoke test (separate results, not research numbers).'
& $PythonExe run_pipeline.py --smoke
if ($LASTEXITCODE -ne 0) { throw 'Smoke test failed; full experiment was not started. See the error above.' }
if (-not $SmokeOnly) {
    Write-Host 'Stage 2/2: full fixed-protocol experiment.'
    & $PythonExe run_pipeline.py
    if ($LASTEXITCODE -ne 0) { throw 'Experiment interrupted. Rerun with -SkipInstall to resume.' }
}
Write-Host 'Finished. See outputs\latest_run.json for the result directory.'
