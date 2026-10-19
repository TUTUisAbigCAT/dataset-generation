# Explicit environment override, local .venv, then optional private runtime config.
$ErrorActionPreference = 'Stop'
$taskPython = $null
if ($env:DATASET_GENERATION_PYTHON) {
    $taskPython = $env:DATASET_GENERATION_PYTHON
} elseif (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')) {
    $taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
} elseif (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'runtime_config.json') -PathType Leaf) {
    $taskRuntime = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'runtime_config.json') -Raw | ConvertFrom-Json
    $taskPython = $taskRuntime.python
}
if (-not $taskPython -or -not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    throw 'Python environment not found. Create .venv with Python 3.12 and install requirement.txt, or set DATASET_GENERATION_PYTHON.'
}
$taskExitCode = 0
Push-Location -LiteralPath $PSScriptRoot
try {
    & $taskPython -B (Join-Path $PSScriptRoot 'scripts\run_dreamt.py') @args
    $taskExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExitCode
