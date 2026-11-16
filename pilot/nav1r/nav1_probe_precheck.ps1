# nav1_probe_precheck.ps1 - is the sim free? + parse-check the probe ps1 files.
$ErrorActionPreference = 'Continue'
$out = 'D:\fpv\ops\nav1_probe_precheck.txt'
"PRECHECK $(Get-Date -Format 'o')" | Out-File $out -Encoding utf8
$blocks = @(Get-Process 'Blocks*' -ErrorAction SilentlyContinue)
"BLOCKS_COUNT=$($blocks.Count)" | Add-Content $out
foreach ($b in $blocks) { "BLOCK pid=$($b.Id)" | Add-Content $out }
try {
  $tasks = @(Get-ScheduledTask | Where-Object { $_.TaskName -like 'FPV*' -or $_.TaskName -like 'CC*' })
  foreach ($t in $tasks) {
    $st = (Get-ScheduledTaskInfo -InputObject $t -ErrorAction SilentlyContinue).State
    "$($t.TaskName)=$st" | Add-Content $out
  }
} catch { "TASKQUERY_ERR $($_.Exception.Message)" | Add-Content $out }
"YOGA_AWAY=$([bool](Test-Path 'D:\fpv\ops\YOGA_AWAY'))" | Add-Content $out
"CHAIN_OWNS_SIM=$([bool](Test-Path 'D:\fpv\ops\CHAIN_OWNS_SIM'))" | Add-Content $out
try {
  $batt = @(Get-CimInstance Win32_Battery -ErrorAction Stop)
  foreach ($x in $batt) { "BATTERY_STATUS=$($x.BatteryStatus) PCT=$($x.EstimatedChargeRemaining)" | Add-Content $out }
} catch { "BATTERY_ERR $($_.Exception.Message)" | Add-Content $out }
try {
  $h = Get-FileHash 'D:\fpv\ops\nav1_simlink.py' -Algorithm SHA256
  "SHASIMLINK=$($h.Hash)" | Add-Content $out
} catch { "SHASIMLINK_ERR $($_.Exception.Message)" | Add-Content $out }
foreach ($f in @('nav1_probe_loop.ps1','nav1_probe_register.ps1')) {
  $p = "D:\fpv\ops\$f"
  if (Test-Path $p) {
    $toks = $null; $errs = $null
    [System.Management.Automation.Language.Parser]::ParseFile($p, [ref]$toks, [ref]$errs) | Out-Null
    if ($errs.Count -eq 0) { "PARSE_OK $f" | Add-Content $out } else { "PARSE_FAIL $f $($errs[0].Message)" | Add-Content $out }
  } else { "PARSE_MISSING $f" | Add-Content $out }
}
try { query user 2>&1 | Out-String | Add-Content $out } catch { "QUERY_USER_ERR" | Add-Content $out }
"PRECHECK_DONE" | Add-Content $out
