# nav1_r_loop.ps1 - NAV1-R SimLink rehearsal rig (FPVNAV1R scheduled task).
# Modeled on D:\fpv\ops\s6_loop.ps1: busy guard -> Blocks launch + port wait ->
# run bridge (runs the whole rehearsal) -> kill ONLY own Blocks PID -> NAV1R_DONE.
# Locked design: NAV1R_RIG_SPEC.md (card t_70db6699).
param()
if (& 'D:\fpv\ops\fpv_busy_guard.ps1' -Self $PSCommandPath) { exit 0 }  # shared busy guard (any other FPV* Running or any Blocks sim -> skip)
$ErrorActionPreference = 'Continue'
$ops = 'D:\fpv\ops'
$status = "$ops\nav1_r_status.txt"
function LogW($m) { "$(Get-Date -Format 'o') NAV1R $m" | Add-Content "$ops\task_log.txt" }
function Set-Status($m) { "$(Get-Date -Format 'o') WRAP $m" | Add-Content $status }
$blocks = $null
LogW "LOOP_STARTED pid=$PID"
try {
  $t0 = Get-Date
  $blocks = Start-Process -FilePath 'D:\fpv\sim\blocks\Blocks.exe' -WorkingDirectory 'D:\fpv\sim\blocks' -PassThru
  LogW "SIM_LAUNCHED pid=$($blocks.Id)"
  $portOk = $false
  while (((Get-Date) - $t0).TotalSeconds -lt 300) {
    $c1 = Test-NetConnection -ComputerName 127.0.0.1 -Port 8989 -WarningAction SilentlyContinue
    $c2 = Test-NetConnection -ComputerName 127.0.0.1 -Port 8990 -WarningAction SilentlyContinue
    if ($c1.TcpTestSucceeded -and $c2.TcpTestSucceeded) { $portOk = $true; break }
    Start-Sleep -Seconds 5
  }
  if (-not $portOk) {
    LogW 'PORT_TIMEOUT_8989_300S'
    Set-Status 'PORT_TIMEOUT_8989_300S'
    & taskkill.exe /PID $blocks.Id /T /F 2>&1 | Out-Null
    LogW 'BLOCKS_STOPPED_PORT_TIMEOUT'
    exit 2
  }
  LogW "PORTS_UP_8989_8990 after $([math]::Round(((Get-Date) - $t0).TotalSeconds, 1))s"
  Set-Status 'PORTS_UP_8989_8990'
  $bridgeArgs = @('D:\fpv\ops\nav1_simlink.py')
  $job = Start-Process -FilePath 'D:\fpv\venv312\Scripts\python.exe' -ArgumentList $bridgeArgs -WorkingDirectory 'D:\fpv\ProjectAirSim\client\python\example_user_scripts' -PassThru -NoNewWindow -RedirectStandardOutput "$ops\nav1_py_stdout.txt" -RedirectStandardError "$ops\nav1_py_stderr.txt"
  LogW "BRIDGE_STARTED pid=$($job.Id)"
  try { $job | Wait-Process -Timeout 14400 -ErrorAction Stop } catch { }
  if (-not $job.HasExited) {
    LogW 'BRIDGE_TIMEOUT_14400S_KILLING_PYTHON'
    & taskkill.exe /PID $job.Id /T /F 2>&1 | Out-Null
  }
  LogW 'BRIDGE_EXITED'
} finally {
  if ($blocks -and -not $blocks.HasExited) {
    & taskkill.exe /PID $blocks.Id /T /F 2>&1 | Out-Null
    LogW "BLOCKS_STOPPED pid=$($blocks.Id)"
  }
  Set-Status 'NAV1R_DONE'
  LogW 'LOOP_DONE'
}
