# register_s6.ps1 - register + fire FPVS6Loop (one-shot S6 loopback run).
# Modeled on register_probeb.ps1 (InteractiveToken, same user, XML create).
# usage: register_s6.ps1 -RunnerHost <SABLE_TAILSCALE_IP> [-StartAt yyyy-MM-ddTHH:mm:ss]
param($RunnerHost, $StartAt)
$ErrorActionPreference = 'Continue'
if (-not $RunnerHost) {
  Write-Output 'usage: register_s6.ps1 -RunnerHost <tailscale-ip> [-StartAt yyyy-MM-ddTHH:mm:ss]'
  exit 2
}
if (-not $StartAt) { $StartAt = (Get-Date).AddMinutes(1).ToString('yyyy-MM-ddTHH:mm:ss') }
$xml = @'
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Date>2026-11-09T00:00:00</Date>
    <Author>LAPTOP\user</Author>
    <URI>\FPVS6Loop</URI>
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
      <Arguments>-NoProfile -ExecutionPolicy Bypass -File D:\fpv\ops\s6_loop.ps1 -RunnerHost RUNNERHOST -RunnerPort 9931</Arguments>
    </Exec>
  </Actions>
</Task>
'@
$xml = $xml -replace 'SUPERSTART', $StartAt
$xml = $xml -replace 'RUNNERHOST', $RunnerHost
$xml | Out-File -Encoding unicode D:\fpv\ops\fpvs6loop.xml
schtasks /delete /tn FPVS6Loop /f 2>$null | Out-Null
schtasks /create /tn FPVS6Loop /xml D:\fpv\ops\fpvs6loop.xml /f
Write-Output "REGISTERED FPVS6Loop boundary=$StartAt runner=$RunnerHost"
schtasks /query /tn FPVS6Loop /fo LIST
