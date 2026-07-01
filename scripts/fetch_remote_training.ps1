param(
    [Parameter(Mandatory = $true)]
    [string]$SshTarget,

    [Parameter(Mandatory = $true)]
    [string]$RemoteJobDir,

    [string]$Profile = "default",
    [string]$Workspace = "."
)

$ErrorActionPreference = "Stop"

$WorkspacePath = Resolve-Path $Workspace
$ProfileDir = Join-Path $WorkspacePath "profiles\$Profile"
New-Item -ItemType Directory -Path $ProfileDir -Force | Out-Null

$ModelPath = Join-Path $ProfileDir "model.onnx"
$MetadataPath = Join-Path $ProfileDir "model_metadata.json"
$MetricsPath = Join-Path $ProfileDir "metrics.json"

scp "${SshTarget}:$RemoteJobDir/outputs/model.onnx" $ModelPath
scp "${SshTarget}:$RemoteJobDir/outputs/model_metadata.json" $MetadataPath
scp "${SshTarget}:$RemoteJobDir/outputs/metrics.json" $MetricsPath

Write-Host "Fetched:"
Write-Host "  $ModelPath"
Write-Host "  $MetadataPath"
Write-Host "  $MetricsPath"
Write-Host "Run locally with:"
Write-Host ".\.venv\Scripts\eye-cursor.exe run --profile $Profile"
