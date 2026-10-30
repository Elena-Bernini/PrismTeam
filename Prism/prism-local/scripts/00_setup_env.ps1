param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$folders = @(
    "config",
    "data/raw/rplan/extracted",
    "data/processed/rplan",
    "scripts",
    "src",
    "outputs/dataset_audit",
    "outputs/retrieval",
    "outputs/partial_layout_retrieval",
    "outputs/evaluation_resnet50",
    "outputs/evaluation_dinov2",
    "outputs/evaluation_vit",
    "outputs/evaluation_comparison",
    "docs/pipeline"
)
foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
}

$version = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
if (-not $version.StartsWith("3.11")) {
    Write-Warning "Recommended Python is 3.11.x for parity with AImageLab-HPC. Detected: $version"
}

Write-Host "Folder skeleton ready."
Write-Host "Next steps:"
Write-Host "  $Python -m venv .venv"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  pip install -r requirements.txt"
Write-Host "  # install CUDA PyTorch separately according to your local CUDA version"
