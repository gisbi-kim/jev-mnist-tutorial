param([int]$Steps = 600, [int]$TrainImages = 1200, [int]$TestImages = 100)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot '.tmp') | Out-Null
$env:TEMP = Join-Path $PSScriptRoot '.tmp'
$env:TMP = $env:TEMP
function Invoke-Checked {
    param([string]$Executable, [string[]]$CommandArgs)
    & $Executable @CommandArgs
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    Invoke-Checked 'python' @('-m','venv','.venv')
}
$TaskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $TaskPython -m pip --version
if ($LASTEXITCODE -ne 0) {
    Invoke-Checked $TaskPython @('-m','ensurepip','--upgrade','--default-pip')
}
$env:HF_HOME = Join-Path $PSScriptRoot '.cache\huggingface'
$env:TORCH_HOME = Join-Path $PSScriptRoot '.cache\torch'
Invoke-Checked $TaskPython @('-m','pip','install','--upgrade','pip')
# CUDA 13 wheels follow the current official Unsloth Windows guide.
Invoke-Checked $TaskPython @('-m','pip','install','torch','torchvision','--index-url','https://download.pytorch.org/whl/cu130')
Invoke-Checked $TaskPython @('-m','pip','install','-r','requirements-train.txt')
Invoke-Checked $TaskPython @('-c','import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name()); print((torch.ones(2,device="cuda")+1).cpu())')
Invoke-Checked $TaskPython @('-c','from unsloth import FastDecisionModel; import inspect; assert "images" in inspect.signature(FastDecisionModel.predict).parameters, "Update Unsloth: multimodal decision API is required"')
Invoke-Checked $TaskPython @('data.py')
Invoke-Checked $TaskPython @('-m','unittest','discover','-s','tests','-v')
Invoke-Checked $TaskPython @('train.py','--steps',"$Steps",'--train-images',"$TrainImages",'--test-images',"$TestImages")
Write-Host 'Training complete. GUI: http://127.0.0.1:7864'
Start-Process 'http://127.0.0.1:7864'
Invoke-Checked $TaskPython @('server.py')
