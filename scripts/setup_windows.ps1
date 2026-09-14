param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if ($Python) {
    $Bootstrap = @($Python)
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.11 -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $Bootstrap = @("py", "-3.11")
    } else {
        & py -3.12 -c "import sys" 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Python 3.11 or 3.12 x64 is required."
        }
        $Bootstrap = @("py", "-3.12")
    }
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $Bootstrap = @("python")
} else {
    throw "Python 3.11 or 3.12 is required. Install 64-bit Python, then rerun this script."
}

$BootstrapExe = $Bootstrap[0]
$BootstrapArgs = @()
if ($Bootstrap.Count -gt 1) {
    $BootstrapArgs = $Bootstrap[1..($Bootstrap.Count - 1)]
}
& $BootstrapExe @BootstrapArgs -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) and sys.maxsize > 2**32 else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Selected interpreter must be 64-bit Python 3.11 or 3.12: $($Bootstrap -join ' ')"
}
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    & $BootstrapExe @BootstrapArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv" }
}
& $VenvPython -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) and sys.maxsize > 2**32 else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Existing .venv is incompatible. Remove it and rerun setup_windows.ps1."
}
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip" }
& $VenvPython -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw "Failed to install locked dependencies" }
& $VenvPython -m pip install --no-deps --no-build-isolation -e .
if ($LASTEXITCODE -ne 0) { throw "Failed to install the benchmark package" }
& $VenvPython -m go2_mujoco_benchmark.doctor --allow-missing-policy
if ($LASTEXITCODE -ne 0) { throw "Environment doctor failed" }
Write-Host "WINDOWS_SETUP_OK"
