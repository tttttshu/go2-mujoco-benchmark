param(
    [string]$Track = "configs/tracks/flat_20m.yaml",
    [string]$PolicyDir = "policies/him_policy",
    [double]$Vx = 1.0,
    [double]$Vy = 0.0,
    [double]$Wz = 0.0,
    [int]$Port = 8080,
    [ValidateRange(0, 9)]
    [int]$DifficultyLevel = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run scripts\setup_windows.ps1 first."
}
Set-Location $ProjectRoot
$ViewerArgs = @(
    "-m", "go2_mujoco_benchmark.viewer",
    "--policy-dir", $PolicyDir,
    "--robot-scene", "third_party/unitree_go2/scene.xml",
    "--track", $Track,
    "--backend", "mjviser",
    "--host", "127.0.0.1",
    "--port", $Port,
    "--vx", $Vx, "--vy", $Vy, "--wz", $Wz,
    "--open-browser"
)
if ($DifficultyLevel -gt 0) {
    $ViewerArgs += @("--difficulty-level", $DifficultyLevel)
}
& $Python @ViewerArgs
if ($LASTEXITCODE -ne 0) { throw "Windows viewer exited with code $LASTEXITCODE" }
