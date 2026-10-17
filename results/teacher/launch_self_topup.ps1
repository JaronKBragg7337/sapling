$bat = "C:\Users\lilli\Projects\sapling\results\teacher\run_self_topup.bat"
$p = Start-Process -FilePath $bat -WindowStyle Minimized -PassThru
Write-Output "started pid $($p.Id)"
Start-Sleep -Seconds 20
Get-Content "C:\Users\lilli\Projects\sapling\results\teacher\self_topup_stdout.txt" -ErrorAction SilentlyContinue
