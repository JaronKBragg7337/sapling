$bat = "C:\Users\lilli\Projects\sapling\results\teacher\run_fold_s23.bat"
$p = Start-Process -FilePath $bat -WindowStyle Minimized -PassThru
Write-Output "started pid $($p.Id)"
Start-Sleep -Seconds 20
$found = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*fold_test*' }
if ($found) { $found | ForEach-Object { Write-Output "fold_test running: pid $($_.ProcessId)" } } else { Write-Output "NO fold_test python found" }
