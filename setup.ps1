param([switch]$Notebook)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv .venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv .venv
    } else {
        throw 'Install Python 3.12, then run setup.ps1 again.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed. Install Python 3.12.' }
}

$requirements = if ($Notebook) { 'requirements-notebook.txt' } else { 'requirements.txt' }
& '.\.venv\Scripts\python.exe' -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if ($Notebook) {
    & '.\.venv\Scripts\python.exe' -m ipykernel install --prefix '.venv' --name 'digits-rf' --display-name 'Python (Digits RF)'
    if ($LASTEXITCODE -ne 0) { throw 'Kernel installation failed.' }
}
Write-Host 'Setup complete.'
Write-Host 'Demo: .\.venv\Scripts\python.exe demo.py --index 123'
Write-Host 'Verify: .\.venv\Scripts\python.exe -m src.validation'
Write-Host 'New experiment: .\.venv\Scripts\python.exe run_experiment.py --output results/reproduction-01'
