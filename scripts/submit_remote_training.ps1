param(
    [Parameter(Mandatory = $true)]
    [string]$SshTarget,

    [Parameter(Mandatory = $true)]
    [string]$RemoteRoot,

    [string]$Profile = "default",
    [string]$Workspace = ".",
    [string]$Calibration = "",
    [string]$JobName = "",
    [switch]$NoSubmit
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path $PSScriptRoot\..
$WorkspacePath = Resolve-Path $Workspace
$ProfileDir = Join-Path $WorkspacePath "profiles\$Profile"

if (-not $Calibration) {
    $Marker = Join-Path $ProfileDir "latest_calibration.txt"
    if (-not (Test-Path $Marker)) {
        throw "No latest calibration marker found at $Marker. Run calibration first or pass -Calibration."
    }
    $CalibrationName = (Get-Content $Marker -Raw).Trim()
    $Calibration = Join-Path $ProfileDir $CalibrationName
}

$CalibrationPath = Resolve-Path $Calibration

if (-not $JobName) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $JobName = "eye_cursor_${Profile}_$Stamp"
}

$StageRoot = Join-Path $ProjectRoot ".remote_jobs\$JobName"
$RemoteJobDir = "$RemoteRoot/$JobName"

if (Test-Path $StageRoot) {
    Remove-Item -LiteralPath $StageRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $StageRoot | Out-Null
New-Item -ItemType Directory -Path (Join-Path $StageRoot "slurm") | Out-Null

Copy-Item -Path (Join-Path $ProjectRoot "src") -Destination (Join-Path $StageRoot "src") -Recurse
Copy-Item -Path (Join-Path $ProjectRoot "pyproject.toml") -Destination $StageRoot
Copy-Item -Path (Join-Path $ProjectRoot "README.md") -Destination $StageRoot
Copy-Item -Path (Join-Path $ProjectRoot "requirements-server.txt") -Destination $StageRoot
Copy-Item -Path (Join-Path $ProjectRoot "slurm\train_eye_cursor.sbatch") -Destination (Join-Path $StageRoot "train_eye_cursor.sbatch")
Copy-Item -Path $CalibrationPath -Destination (Join-Path $StageRoot "calibration.npz")

Write-Host "Staged job: $StageRoot"
Write-Host "Remote target: ${SshTarget}:$RemoteJobDir"

ssh $SshTarget "mkdir -p '$RemoteJobDir'"
scp -r `
    "$StageRoot\src" `
    "$StageRoot\pyproject.toml" `
    "$StageRoot\README.md" `
    "$StageRoot\requirements-server.txt" `
    "$StageRoot\train_eye_cursor.sbatch" `
    "$StageRoot\calibration.npz" `
    "${SshTarget}:$RemoteJobDir/"

if ($NoSubmit) {
    Write-Host "Uploaded but did not submit because -NoSubmit was set."
    exit 0
}

ssh $SshTarget "cd '$RemoteJobDir' && sbatch train_eye_cursor.sbatch"
Write-Host "Submitted. Later fetch results with:"
Write-Host ".\scripts\fetch_remote_training.ps1 -SshTarget $SshTarget -RemoteJobDir $RemoteJobDir -Profile $Profile"
