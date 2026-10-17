@echo off
cd /d C:\Users\lilli\Projects\sapling
set PY=C:\Users\lilli\scoop\apps\python312\current\python.exe
set PATH=%PATH%;C:\Program Files\GitHub CLI
set LOG=C:\Users\lilli\Projects\sapling\results\cloud_run_20261016_second.log
echo === cloud lesson 2 resume (bundle reused) %DATE% %TIME% === >> "%LOG%"
"%PY%" tools\runpod_run.py >> "%LOG%" 2>&1
echo === runner exited %DATE% %TIME% === >> "%LOG%"
"%PY%" tools\runpod_run.py --reap >> "%LOG%" 2>&1
echo === ALL DONE %DATE% %TIME% === >> "%LOG%"
