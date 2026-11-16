# nav1_probe_register.ps1 - register FPVNAV1Probe (one-shot stall-probe run).
# Cloned from nav1_r_register.ps1 (InteractiveToken, same user, XML create).
# usage: nav1_probe_register.ps1 [-StartAt yyyy-MM-ddTHH:mm:ss]
param($StartAt)
$ErrorActionPreference = 'Continue'
if (-not $StartAt) { $StartAt = (Get-Date).AddMinutes(1).ToString('yyyy-MM-ddTHH:mm:ss') }
$xml = @'
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Date>2026-11-15T00:00:00</Date>
    <Author>laptop\user</Author>
    <URI>\FPVNAV1Probe</URI>
  </RegistrationInfo>
  <Principals>
    <Principal id="Author">
      <UserId>S-1-5-21-1012949973-2203610850-1594816012-1001</UserId>
      <LogonType>InteractiveToken</LogonType>
    </Principal>
  </Principals>
  <Settings>
    <DisallowStartIfOnBatteries>true</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>true</StopIfGoingOnBatteries>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <IdleSettings>
      <Duration>PT10M</Duration>
      <WaitTimeout>PT1H</WaitTimeout>
      <StopOnIdleEnd>true</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
  </Settings>
  <Triggers>
    <TimeTrigger>
      <StartBoundary>SUPERSTART</StartBoundary>
    </TimeTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>powershell</Command>
      <Arguments>-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File D:\fpv\ops\nav1_probe_loop.ps1</Arguments>
    </Exec>
  </Actions>
</Task>
'@
$xml = $xml -replace 'SUPERSTART', $StartAt
$xml | Out-File -Encoding unicode D:\fpv\ops\fpvnav1probe.xml
schtasks /delete /tn FPVNAV1Probe /f 2>$null | Out-Null
schtasks /create /tn FPVNAV1Probe /xml D:\fpv\ops\fpvnav1probe.xml /f
Write-Output "REGISTERED FPVNAV1Probe boundary=$StartAt"
schtasks /query /tn FPVNAV1Probe /fo LIST
