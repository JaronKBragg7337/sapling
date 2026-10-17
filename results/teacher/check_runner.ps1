$found = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*runpod_run*' }
if ($found) { $found | ForEach-Object { Write-Output "runner alive: pid $($_.ProcessId)" } } else { Write-Output "RUNNER NOT RUNNING" }
$bat = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" | Where-Object { $_.CommandLine -like '*run_cloud2*' }
if ($bat) { $bat | ForEach-Object { Write-Output "bat alive: pid $($_.ProcessId)" } } else { Write-Output "bat not running" }
