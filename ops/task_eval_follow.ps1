if (& 'D:\fpv\ops\fpv_busy_guard.ps1' -Self $PSCommandPath) { exit 0 }  # Claude Code 2026-11-11: shared busy guard (any other FPV* Running or any Blocks sim -> skip)
# task_eval_follow.ps1 - S7 follow eval launch rig: starts Blocks, waits for the
# AirSim port, runs rl_eval_follow.py, reports the eval_result.txt tail. Mirrors
# the proven task_eval_v4.ps1 FPV_WAY branch semantics. Deliberately does NOT
# set FPV_POLICY: missing policy_final.zip => rl_eval_follow.py runs in
# baseline-only mode (zero + ibvs, verdict=BASELINE_RECORDED). FPV_DISTURB_*
# vars stay out: the follow env does its own gusts. PS 5.1.
$ErrorActionPreference = 'Continue'
$env:FPV_RUN = 'D:\fpv\rl\runs\follow_v1'
New-Item -ItemType Directory -Force $env:FPV_RUN | Out-Null
$ops = 'D:\fpv\ops'
$wlog = "$ops\eval_follow_out.txt"
function LogW($m) { "$(Get-Date -Format 'o') $m" | Add-Content "$ops\task_log.txt" }
LogW "EVALFOLLOWTASK_STARTED pid=$PID"
# COLLISION GUARD: never share the sim with another FPV task (proven collision 00:50).
if (Get-Process Blocks* -Force -EA SilentlyContinue) {
  LogW 'EVALFOLLOWTASK_SKIPPED_BUSY Blocks already running'
  'SKIPPED_BUSY' | Set-Content $wlog
  'TASK_DONE' | Add-Content $wlog
  exit 0
}
foreach ($t in 'FPVVideo','FPVRL','FPVRLVideo','FPVRLVideoObs','FPVTrain','FPVTrainV3','FPVTrainV4','FPVEval','FPVEvalV4','FPVS2Loopback','FPVYard','FPVWPRoute') {
  try {
    if ((Get-ScheduledTask -TaskName $t -EA Stop).State -eq 'Running') {
      LogW "EVALFOLLOWTASK_SKIPPED_BUSY $t running"
      'SKIPPED_BUSY' | Set-Content $wlog
      'TASK_DONE' | Add-Content $wlog
      exit 0
    }
  } catch { }
}
try {
  $t0 = Get-Date
  Start-Process -FilePath 'D:\fpv\sim\blocks\Blocks.exe' -WorkingDirectory 'D:\fpv\sim\blocks'
  LogW 'SIM_LAUNCHED'
  $portOk = $false
  while (((Get-Date) - $t0).TotalSeconds -lt 300) {
    $c = Test-NetConnection -ComputerName 127.0.0.1 -Port 8989 -WarningAction SilentlyContinue
    if ($c.TcpTestSucceeded) { $portOk = $true; break }
    Start-Sleep -Seconds 5
  }
  if (-not $portOk) {
    LogW 'EVALFOLLOWPORT_TIMEOUT_300s'
    'PORT_TIMEOUT_8989_300S' | Set-Content $wlog
    Stop-Process -Name 'Blocks*' -Force -EA SilentlyContinue
    exit 2
  }
  LogW "PORT_8989_UP after $([math]::Round(((Get-Date) - $t0).TotalSeconds, 1))s"
  $evalStart = Get-Date
  $job = Start-Process -FilePath 'D:\fpv\venv312\Scripts\python.exe' -ArgumentList 'D:\fpv\rl\rl_eval_follow.py' -WorkingDirectory 'D:\fpv\ProjectAirSim\client\python\example_user_scripts' -RedirectStandardOutput "$ops\eval_follow_out.txt" -RedirectStandardError "$ops\eval_follow_err.txt" -PassThru -NoNewWindow
  LogW "EVALFOLLOW_STARTED pid=$($job.Id)"
  try { $job | Wait-Process -Timeout 2400 -ErrorAction Stop } catch { }
  if (-not $job.HasExited) {
    LogW 'EVALFOLLOW_TIMEOUT_2400S_KILLING_PYTHON'
    & taskkill.exe /PID $job.Id /T /F 2>&1 | Out-Null
  }
  # success = python exited clean (or ExitCode unavailable in PS 5.1), stdout has no
  # EVAL_FAILED marker, AND eval_result.txt is FRESH (written after this run started).
  # A previous run's result file must never read as OK (false-OK incident 2026-11-11).
  $reason = ''
  $code = $null
  try { $code = $job.ExitCode } catch { }
  if ($null -ne $code -and $code -ne 0) { $reason = 'exit_code' }
  if (-not $reason -and (Select-String -Path "$ops\eval_follow_out.txt" -Pattern 'EVAL_FAILED' -Quiet -EA SilentlyContinue)) { $reason = 'eval_failed_marker' }
  $tail = ''
  if (-not $reason) {
    $res = 'D:\fpv\rl\runs\follow_v1\eval_result.txt'
    $resultFile = Get-Item $res -EA SilentlyContinue
    if ($resultFile -and $resultFile.LastWriteTime -gt $evalStart.AddSeconds(-5)) {
      $tail = (Get-Content $res -Tail 1 -EA SilentlyContinue)
    }
    if (-not $tail) { $reason = 'stale_result' }
  }
  if ($reason) { LogW "EVALFOLLOW_FAILED reason=$reason" } else { LogW "EVALFOLLOW_OK result_tail=$tail" }
} finally {
  Stop-Process -Name 'Blocks*' -Force -EA SilentlyContinue
  LogW 'BLOCKS_STOPPED'
  Start-Sleep -Seconds 2
  'TASK_DONE' | Add-Content $wlog
  LogW 'EVALFOLLOWTASK_DONE_WRITTEN'
}
