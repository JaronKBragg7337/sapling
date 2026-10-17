@echo off
cd /d C:\Users\lilli\Projects\sapling
set PY=C:\Users\lilli\scoop\apps\python312\current\python.exe
set LOG=C:\Users\lilli\Projects\sapling\results\teacher\self_topup_stdout.txt
echo === stop heartbeat %DATE% %TIME% === > "%LOG%"
"%PY%" tools\gpu_yield.py stop "self-lesson top-up" >> "%LOG%" 2>&1
echo === train start %DATE% %TIME% === >> "%LOG%"
"%PY%" -m sapling.talk train --self-only --epochs 16 --core results/core_checkpoint.pt >> "%LOG%" 2>&1
if errorlevel 1 (
  echo === TRAIN FAILED - no swap, restarting heartbeat === >> "%LOG%"
  "%PY%" tools\gpu_yield.py start "self-lesson top-up" >> "%LOG%" 2>&1
  exit /b 1
)
echo === swap %DATE% %TIME% === >> "%LOG%"
"%PY%" -c "from sapling.grow import swap; swap(staged='results/core_self_topup_staged.pt')" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo === SWAP FAILED - restarting heartbeat === >> "%LOG%"
  "%PY%" tools\gpu_yield.py start "self-lesson top-up" >> "%LOG%" 2>&1
  exit /b 1
)
echo === restart heartbeat %DATE% %TIME% === >> "%LOG%"
"%PY%" tools\gpu_yield.py start "self-lesson top-up" >> "%LOG%" 2>&1
echo === ALL DONE %DATE% %TIME% === >> "%LOG%"
