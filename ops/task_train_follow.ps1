# task_train_follow.ps1 - S7 follow_v1 SAC training rig (mirrors task_train_v4.ps1 semantics)
$ErrorActionPreference = 'Continue'
$ops = 'D:\fpv\ops'
$log = "$ops\task_log.txt"
$status = 'D:\fpv\ops\train_status.txt'
function Log($m) { "$(Get-Date -Format 'o') $m" | Add-Content $log }
"TASK_STARTED $(Get-Date -Format 'o')" | Set-Content "$ops\followtrain_run_out.txt"
Log "FOLLOWTRAIN_STARTED pid=$PID"
# S7 follow_v1: 6 parallel Blocks sims, ports 8989+2k/8990+2k; FPV_STEPS is
# GLOBAL steps across envs; the trainer heartbeats $status (train_status.txt)
# and appends TRAIN_DONE there on clean finish (verified in rl_train_follow_multi.py).
$env:FPV_RUN = 'D:\fpv\rl\runs\follow_v1'
$env:FPV_STEPS = '30000'
$env:FPV_CKPT = '2500'
$env:FPV_MULTI = '6'
$env:FPV_TARGET_SPEED_SCALE = '0.3'   # MUST be 0.3: scale-1.0 baseline was physically un-followable (targets outrun the +-1 m/s drone; measured + logged)
New-Item -ItemType Directory -Force $env:FPV_RUN | Out-Null
# COLLISION GUARD
$tp = 8989; $sp = 8990
if ($env:FPV_PORT_TOPICS) { $tp = [int]$env:FPV_PORT_TOPICS; $sp = $tp + 1 }
$portsBusy = ((Test-NetConnection -ComputerName 127.0.0.1 -Port $tp -WarningAction SilentlyContinue).TcpTestSucceeded) -or ((Test-NetConnection -ComputerName 127.0.0.1 -Port $sp -WarningAction SilentlyContinue).TcpTestSucceeded)
$mine = @()
if ($portsBusy) {
  Log "FOLLOWTRAIN_SKIPPED_BUSY ports $tp/$sp busy"
  'SKIPPED_BUSY' | Add-Content "$ops\followtrain_run_out.txt"
  exit 0
}
foreach ($t in 'FPVVideo','FPVRL','FPVEval','FPVEvalV3','FPVEvalV4','FPVRLVideo','FPVProbe','FPVProbeB','FPVYard','FPVS7Eval','FPVWPRoute') {
  try {
    if ((Get-ScheduledTask -TaskName $t -EA Stop).State -eq 'Running') {
      Log "FOLLOWTRAIN_SKIPPED_BUSY $t running"
      'SKIPPED_BUSY' | Add-Content "$ops\followtrain_run_out.txt"
      exit 0
    }
  } catch { }
}
function Start-Sim {
  $t0 = Get-Date
  $n = 1
  if ($env:FPV_MULTI) { $n = [int]$env:FPV_MULTI }
  $bfr = @(Get-Process -Name 'Blocks*' -EA SilentlyContinue | ForEach-Object { $_.Id })
  $bp = $null
  if ($n -gt 1) {
    # multi-sim: N Blocks instances, env k uses topics 8989+2k / services 8990+2k
    for ($k = 0; $k -lt $n; $k++) {
      $kp = 8989 + 2 * $k; $kq = 8990 + 2 * $k
      Start-Process -FilePath 'D:\fpv\sim\blocks\Blocks.exe' -WorkingDirectory 'D:\fpv\sim\blocks' -ArgumentList "-topicsport=$kp -servicesport=$kq -windowed -ResX=640 -ResY=360 -ExecCmds=`"t.MaxFPS 30`"" -EA SilentlyContinue | Out-Null
    }
  } else {
    $bp = Start-Process -FilePath 'D:\fpv\sim\blocks\Blocks.exe' -WorkingDirectory 'D:\fpv\sim\blocks' -ArgumentList "-topicsport=$script:tp -servicesport=$script:sp -windowed -ResX=640 -ResY=360 -ExecCmds=`"t.MaxFPS 30`"" -PassThru
  }
  $new = @(if ($bp) { $bp.Id })
  # PID diff AFTER all N launches: bootstrapper spawns a Blocks-Win64-Shipping child
  $new += @(Get-Process -Name 'Blocks*' -EA SilentlyContinue | Where-Object { $bfr -notcontains $_.Id } | ForEach-Object { $_.Id })
  $script:mine = @((@($script:mine) + $new) | Sort-Object -Unique)
  Log "FOLLOWTRAIN_SIM_LAUNCHED n=$n BLOCKS_MINE=$($script:mine -join ',')"
  if ($n -gt 1) {
    # poll EVERY topics port (pair is up when topics answers), 300 s total
    while (((Get-Date) - $t0).TotalSeconds -lt 300) {
      $up = 0
      for ($k = 0; $k -lt $n; $k++) {
        if ((Test-NetConnection -ComputerName 127.0.0.1 -Port (8989 + 2 * $k) -WarningAction SilentlyContinue).TcpTestSucceeded) { $up++ }
      }
      if ($up -ge $n) { return $true }
      Start-Sleep -Seconds 5
    }
    return $false
  }
  while (((Get-Date) - $t0).TotalSeconds -lt 300) {
    if ((Test-NetConnection -ComputerName 127.0.0.1 -Port $script:tp -WarningAction SilentlyContinue).TcpTestSucceeded) { return $true }
    Start-Sleep -Seconds 5
  }
  return $false
}
try {
  if (-not (Start-Sim)) {
    Log 'FOLLOWTRAIN_PORT_TIMEOUT_300s'
    'PORT_TIMEOUT' | Add-Content "$ops\followtrain_run_out.txt"
    foreach ($mid in $mine) { Stop-Process -Id $mid -Force -EA SilentlyContinue }
    exit 2
  }
  Log 'FOLLOWTRAIN_PORT_UP'
  $t0 = Get-Date
  $attempt = 1
  $maxAttempts = 5
  $done = $false
  while ($attempt -le $maxAttempts -and -not $done) {
    # S7 follow_v1 trainer (resume-aware per its own header; fixed target unlike
    # v4 which selected among three trainers by FPV_MULTI/FPV_RESID).
    $trainPy = 'D:\fpv\rl\rl_train_follow_multi.py'
    $job = Start-Process -FilePath 'D:\fpv\venv312\Scripts\python.exe' -ArgumentList $trainPy -WorkingDirectory 'D:\fpv\ProjectAirSim\client\python\example_user_scripts' -RedirectStandardOutput "$ops\rlfollow_out.txt" -RedirectStandardError "$ops\rlfollow_err.txt" -PassThru -NoNewWindow
    Log "FOLLOWTRAIN_PY_STARTED attempt=$attempt pid=$($job.Id)"
    $lastBeat = Get-Date
    $stall = $false
    while ($true) {
      Start-Sleep -Seconds 60
      if ($job.HasExited) { break }
      if (Test-Path $status) {
        $mt = (Get-Item $status).LastWriteTime
        if ($mt -gt $lastBeat) { $lastBeat = $mt }
      }
      # pause-aware: a deliberate freeze (D:\fpv\multi\pause_log.txt last line PAUSED) is not a stall
      try { if ((Get-Content 'D:\fpv\multi\pause_log.txt' -Tail 1 -EA Stop) -match 'PAUSED') { $lastBeat = Get-Date } } catch { }
      if (((Get-Date) - $lastBeat).TotalSeconds -gt 600) { $stall = $true; break }
      if (((Get-Date) - $t0).TotalSeconds -gt 25200) { Log 'FOLLOWTRAIN_GLOBAL_TIMEOUT_7H'; break }
    }
    # SUCCESS CHECK via status file: Start-Process -PassThru ExitCode is $null in
    # PS 5.1, so the old `$job.ExitCode -eq 0` test NEVER fired and clean TRAIN_DONE
    # exits were misclassified as crashes -> zombie auto-restart (incident 08:49, 11-08).
    if ($job.HasExited -and (Test-Path $status) -and (((Get-Content $status | Select-Object -Last 1) + '') -match 'TRAIN_DONE')) {
      Log "FOLLOWTRAIN_EXITED_OK attempt=$attempt (TRAIN_DONE seen)"
      $done = $true
      break
    }
    Log "FOLLOWTRAIN_ATTEMPT_END attempt=$attempt exited=$($job.HasExited) code=$(if($job.HasExited){$job.ExitCode}else{'running'}) stall=$stall"
    if (-not $job.HasExited) { Stop-Process -Id $job.Id -Force -EA SilentlyContinue }
    Start-Sleep -Seconds 5
    # sim may be wedged too: recycle ONLY my own instance between attempts
    foreach ($mid in $mine) { Stop-Process -Id $mid -Force -EA SilentlyContinue }
    $mine = @()
    Start-Sleep -Seconds 15
    if (-not (Start-Sim)) { Log 'FOLLOWTRAIN_RESTART_PORT_TIMEOUT'; break }
    $attempt++
  }
  if (-not $done) { Log "FOLLOWTRAIN_FAILED_AFTER_ATTEMPTS attempts=$attempt" }
} finally {
  foreach ($mid in $mine) { Stop-Process -Id $mid -Force -EA SilentlyContinue }
  Log 'BLOCKS_STOPPED'
  'TASK_DONE' | Add-Content "$ops\followtrain_run_out.txt"
  Log 'FOLLOWTRAIN_DONE_WRITTEN'
}
