$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$TaskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $TaskPython)) { $TaskPython = 'python' }
Start-Process 'http://127.0.0.1:7864'
& $TaskPython server.py
