# s6_loop.ps1 - S6 phone-pilot loopback eval rig (FPVS6Loop scheduled task).
# Modeled on D:\fpv\rl\task_eval_v4.ps1 (FPV_WAY branch): collision guard ->
# Blocks launch + port wait -> run bridge -> teardown Blocks -> result tail.
param($RunnerHost, $RunnerPort = 9931)
$ErrorActionPreference = 'Continue'
$ops = 'D:\fpv\ops'
$out = "$ops\s6_loop_out.txt"
function LogW($m) { "$(Get-Date -Format 'o') $m" | Add-Content "$ops\task_log.txt" }
LogW "S6LOOP_STARTED pid=$PID runner=${RunnerHost}:${RunnerPort}"
if (-not $RunnerHost) {
  LogW 'S6LOOP_NO_RUNNERHOST'
  'NO_RUNNERHOST' | Set-Content $out
  'TASK_DONE' | Add-Content $out
  exit 2
}
# COLLISION GUARD: never share the sim with Blocks or another FPV task.
if (Get-Process Blocks* -Force -EA SilentlyContinue) {
  LogW 'S6LOOP_SKIPPED_BUSY Blocks already running'
  'SKIPPED_BUSY' | Set-Content $out
  'TASK_DONE' | Add-Content $out
  exit 0
}
foreach ($t in 'FPVTrainV4','FPVEvalV4','FPVRL') {
  try {
    if ((Get-ScheduledTask -TaskName $t -EA Stop).State -eq 'Running') {
      LogW "S6LOOP_SKIPPED_BUSY $t running"
      'SKIPPED_BUSY' | Set-Content $out
      'TASK_DONE' | Add-Content $out
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
    LogW 'S6LOOP_PORT_TIMEOUT_300s'
    'PORT_TIMEOUT_8989_300S' | Set-Content $out
    exit 2
  }
  LogW "PORT_8989_UP after $([math]::Round(((Get-Date) - $t0).TotalSeconds, 1))s"
  $bridgeArgs = @('D:\fpv\ops\loopback_bridge.py', '--runner-host', "$RunnerHost", '--runner-port', "$RunnerPort")
  $job = Start-Process -FilePath 'D:\fpv\venv312\Scripts\python.exe' -ArgumentList $bridgeArgs -PassThru -NoNewWindow -RedirectStandardOutput "$ops\s6_py_stdout.txt" -RedirectStandardError "$ops\s6_py_stderr.txt"
  LogW "BRIDGE_STARTED pid=$($job.Id)"
  try { $job | Wait-Process -Timeout 2400 -ErrorAction Stop } catch { }
  if (-not $job.HasExited) {
    LogW 'BRIDGE_TIMEOUT_2400S_KILLING_PYTHON'
    Stop-Process -Id $job.Id -Force -EA SilentlyContinue
  }
  LogW "BRIDGE_EXITED"
} finally {
  Stop-Process -Name 'Blocks*' -Force -EA SilentlyContinue
  LogW 'BLOCKS_STOPPED'
  Start-Sleep -Seconds 2
  $res = "$ops\eval_loopback_result.txt"
  if (Test-Path $res) {
    'RESULT_TAIL:' | Add-Content $out
    Get-Content $res -Tail 12 -EA SilentlyContinue | Add-Content $out
  } else {
    'NO_RESULT_FILE' | Add-Content $out
  }
  'TASK_DONE' | Add-Content $out
  LogW 'S6LOOP_DONE_WRITTEN'
}
