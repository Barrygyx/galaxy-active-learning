param([switch]$SkipInstall, [string]$Run = "", [int]$Labels = 0)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Existing .venv not found. Extract these files into galaxy-active-learning alongside outputs and cache."
}
if (-not $SkipInstall) {
    & $Python -m pip install -r (Join-Path $PSScriptRoot "requirements_head_ablation.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}
$Arguments = @((Join-Path $PSScriptRoot "run_head_ablation.py"))
if ($Run) { $Arguments += @("--run", $Run) }
if ($Labels -gt 0) { $Arguments += @("--labels", "$Labels") }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Ablation failed. Original results remain unchanged; read the error above." }
Write-Host "Complete. See outputs\latest_head_ablation.json for the report directory."
