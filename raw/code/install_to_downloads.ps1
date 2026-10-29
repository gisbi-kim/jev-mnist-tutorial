# Run this file yourself in a regular PowerShell terminal after reviewing it.
# It copies this project to a NEW Downloads subfolder; it never overwrites a project.
param([switch]$StartTraining)
$ErrorActionPreference = 'Stop'
$TaskDownloads = Join-Path ([Environment]::GetFolderPath('UserProfile')) 'Downloads'
$TaskDestination = Join-Path $TaskDownloads 'mnist-decision-lab'
if (Test-Path -LiteralPath $TaskDestination) {
    $TaskDestination = Join-Path $TaskDownloads ('mnist-decision-lab-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
New-Item -ItemType Directory -Path $TaskDestination -ErrorAction Stop | Out-Null
Get-ChildItem -LiteralPath $PSScriptRoot -Force |
    Where-Object { $_.Name -notin @('.venv', '__pycache__', '.cache') } |
    Copy-Item -Destination $TaskDestination -Recurse -ErrorAction Stop
Write-Host "Project copied to $TaskDestination"
if ($StartTraining) {
    & (Join-Path $TaskDestination 'setup_and_train.ps1')
}
