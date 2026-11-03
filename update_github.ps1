param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Message
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
$env:GIT_TERMINAL_PROMPT = '0'
Set-Location -LiteralPath $PSScriptRoot

$repositoryRoot = (& git rev-parse --show-toplevel).Replace('/', '\')
if ($LASTEXITCODE -ne 0) { throw 'This project is not a Git repository.' }
if ((Resolve-Path -LiteralPath $repositoryRoot).Path -ne (Resolve-Path -LiteralPath $PSScriptRoot).Path) {
    throw 'Git root does not match the project folder.'
}
& git remote get-url origin
if ($LASTEXITCODE -ne 0) { throw 'Configure the origin remote before pushing.' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Run setup.ps1 first.' }

& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Tests failed. Nothing was committed or pushed.' }
& '.\.venv\Scripts\python.exe' -m src.validation
if ($LASTEXITCODE -ne 0) { throw 'Result validation failed. Nothing was committed or pushed.' }
& '.\.venv\Scripts\python.exe' -m src.robustness_validation --if-exists
if ($LASTEXITCODE -ne 0) { throw 'Development study validation failed. Nothing was committed or pushed.' }
& '.\.venv\Scripts\python.exe' -m src.report_facts --check
if ($LASTEXITCODE -ne 0) { throw 'Report facts disagree with saved logs. Nothing was committed or pushed.' }
& '.\.venv\Scripts\python.exe' -m src.report_evaluation --check
if ($LASTEXITCODE -ne 0) { throw 'Train/CV/test report tables failed validation. Nothing was committed or pushed.' }
& '.\.venv\Scripts\python.exe' demo_optimization.py validate --output results/week9_demo/run-2026-11-01
if ($LASTEXITCODE -ne 0) { throw 'Teaching demo validation failed. Nothing was committed or pushed.' }

for ($attempt = 0; $attempt -lt 3; $attempt++) {
    & git add --all
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Milliseconds 600
}
if ($LASTEXITCODE -ne 0) { throw 'Git staging failed.' }
& git diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    & git commit -m $Message
    if ($LASTEXITCODE -ne 0) { throw 'Git commit failed.' }
} elseif ($LASTEXITCODE -ne 0) {
    throw 'Cannot inspect staged changes.'
} else {
    Write-Host 'No file changes to commit. Checking remote synchronization.'
}
& git push --set-upstream origin HEAD
if ($LASTEXITCODE -ne 0) { throw 'Push failed. The local commit is preserved; fix authentication/network and retry.' }
& git log -1 --format='%h %s'
Write-Host 'GitHub synchronization complete.'
