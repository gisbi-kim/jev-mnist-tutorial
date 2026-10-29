param([int]$Port = 7865)
$ErrorActionPreference = 'Stop'
Write-Host "Tutorial: http://127.0.0.1:$Port/site/index.html"
Write-Host 'Press Ctrl+C to stop. The original training server is separate.'
python -m http.server $Port --bind 127.0.0.1 --directory $PSScriptRoot
