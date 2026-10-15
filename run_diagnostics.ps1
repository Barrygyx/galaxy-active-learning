param([switch]$SkipInstall, [string]$Run = "")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Existing virtual environment missing. Extract this package into galaxy-active-learning, alongside .venv and outputs."
}
if (-not $SkipInstall) {
    & $Python -m pip install -r (Join-Path $PSScriptRoot "requirements_diagnostics.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}
$Arguments = @((Join-Path $PSScriptRoot "diagnose_models.py"))
if ($Run) { $Arguments += @("--run", $Run) }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Diagnosis failed. Read the error above; original experiment is unchanged." }
Write-Host "Diagnosis complete. See outputs\latest_diagnostic.json for the result directory."
