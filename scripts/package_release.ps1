# Create the core DocMirror source archive on Windows without Bash.

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Output = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-PackageLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    Write-Host "[package] $Message"
}

function Resolve-AbsolutePath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$BasePath
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $Path))
}

function Remove-PackageTempDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd('\', '/')
    $resolved = [System.IO.Path]::GetFullPath($Path)
    $expectedPrefix = $tempRoot + [System.IO.Path]::DirectorySeparatorChar
    $leaf = Split-Path -Leaf $resolved
    if (-not $resolved.StartsWith($expectedPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
        -not $leaf.StartsWith("docmirror-package-", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove unexpected temporary directory: $resolved"
    }

    Remove-Item -LiteralPath $resolved -Recurse -Force
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $repoRoot "docmirror-complete.tar.gz"
}
$outputPath = Resolve-AbsolutePath -Path $Output -BasePath (Get-Location).Path
if (-not $outputPath.EndsWith(".tar.gz", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output filename must end in .tar.gz: $outputPath"
}

$requiredEntries = @(
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "LICENSE",
    "Dockerfile",
    "docker-compose.yml",
    "Dockerfile.gpu",
    "docker-compose.gpu.yml",
    "requirements-gpu-cu126.in",
    "requirements-gpu-cu126.txt",
    "docmirror"
)

foreach ($entry in $requiredEntries) {
    $source = Join-Path $repoRoot $entry
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Required release entry is missing: $entry"
    }
}

$tarCommand = (Get-Command tar.exe -ErrorAction Stop).Source
$outputDirectory = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

$packageId = [System.Guid]::NewGuid().ToString("N")
$tempDirectory = Join-Path ([System.IO.Path]::GetTempPath()) "docmirror-package-$packageId"
$stageDirectory = Join-Path $tempDirectory "docmirror"
$archiveTemp = Join-Path $outputDirectory ".docmirror-archive-$packageId.tmp"

try {
    New-Item -ItemType Directory -Path $stageDirectory -Force | Out-Null

    Write-PackageLog "Staging required release files"
    foreach ($entry in $requiredEntries) {
        $source = Join-Path $repoRoot $entry
        $destination = Join-Path $stageDirectory $entry
        Copy-Item -LiteralPath $source -Destination $destination -Recurse -Force
    }

    $reparsePoint = Get-ChildItem -LiteralPath $stageDirectory -Recurse -Force |
        Where-Object { ($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0 } |
        Select-Object -First 1
    if ($null -ne $reparsePoint) {
        throw "Links and reparse points are not allowed in the release: $($reparsePoint.FullName)"
    }

    Write-PackageLog "Creating $outputPath"
    $tarArguments = @(
        "-czf", $archiveTemp,
        "--exclude=*/__pycache__",
        "--exclude=*.pyc",
        "--exclude=*.pyo",
        "--exclude=*/.DS_Store",
        "--exclude=*/.plugin_state.json",
        "--exclude=*/.pytest_cache",
        "--exclude=*/.mypy_cache",
        "--exclude=*/.ruff_cache",
        "--exclude=*/.git",
        "--exclude=*/.git/*",
        "--exclude=*/.env",
        "--exclude=*/credentials",
        "--exclude=*/credentials/*",
        "--exclude=*/secrets",
        "--exclude=*/secrets/*",
        "--exclude=*.log",
        "-C", $tempDirectory,
        "docmirror"
    )
    & $tarCommand @tarArguments
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed with exit code $LASTEXITCODE"
    }

    $archiveListing = @(& $tarCommand -tzf $archiveTemp)
    if ($LASTEXITCODE -ne 0) {
        throw "Created archive failed gzip/tar validation"
    }

    $requiredMembers = @(
        "docmirror/pyproject.toml",
        "docmirror/requirements.txt",
        "docmirror/Dockerfile",
        "docmirror/docker-compose.yml",
        "docmirror/Dockerfile.gpu",
        "docmirror/docker-compose.gpu.yml",
        "docmirror/requirements-gpu-cu126.in",
        "docmirror/requirements-gpu-cu126.txt",
        "docmirror/docmirror/server/api.py",
        "docmirror/docmirror/server/parse_process_manager.py",
        "docmirror/docmirror/server/parse_worker.py"
    )
    foreach ($member in $requiredMembers) {
        if ($archiveListing -notcontains $member) {
            throw "Created archive is missing: $member"
        }
    }

    $commercialMember = $archiveListing |
        Where-Object { $_ -match '^docmirror/(docmirror_enterprise|docmirror_finance)(/|$)' } |
        Select-Object -First 1
    if ($null -ne $commercialMember) {
        throw "Created core archive unexpectedly contains a commercial package: $commercialMember"
    }

    if (Test-Path -LiteralPath $outputPath) {
        Remove-Item -LiteralPath $outputPath -Force
    }
    Move-Item -LiteralPath $archiveTemp -Destination $outputPath -Force

    $sha256 = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-PackageLog "Package completed successfully"
    Write-Host "Archive:  $outputPath"
    Write-Host "SHA-256:  $sha256"
}
finally {
    if (Test-Path -LiteralPath $archiveTemp) {
        Remove-Item -LiteralPath $archiveTemp -Force
    }
    Remove-PackageTempDirectory -Path $tempDirectory
}
