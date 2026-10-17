$bat = "C:\Users\lilli\Projects\sapling\results\teacher\run_cloud2.bat"
$p = Start-Process -FilePath $bat -WindowStyle Minimized -PassThru
Write-Output "started pid $($p.Id)"
Start-Sleep -Seconds 30
Get-Content "C:\Users\lilli\Projects\sapling\results\cloud_run_20261016_second.log" -ErrorAction SilentlyContinue
