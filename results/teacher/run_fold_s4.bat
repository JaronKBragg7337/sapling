@echo off
cd /d C:\Users\lilli\Projects\sapling
set PY=C:\Users\lilli\scoop\apps\python312\current\python.exe
set LOG=C:\Users\lilli\Projects\sapling\results\teacher\fold_s4_stdout.txt
echo === S4 start %DATE% %TIME% === > "%LOG%"
"%PY%" tools\fold_test.py --core results\cloud\20261016-0122\core_staged.pt --label "S4 no-school-docs + 120 steps (D34 lr)" --no-school-docs --steps 120 >> "%LOG%" 2>&1
echo === ALL DONE %DATE% %TIME% === >> "%LOG%"
