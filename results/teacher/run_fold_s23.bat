@echo off
cd /d C:\Users\lilli\Projects\sapling
set PY=C:\Users\lilli\scoop\apps\python312\current\python.exe
set LOG=C:\Users\lilli\Projects\sapling\results\teacher\fold_s23_stdout.txt
echo === S2 start %DATE% %TIME% === > "%LOG%"
"%PY%" tools\fold_test.py --core results\cloud\20261016-0122\core_staged.pt --label "S2 no-school-docs (D34 lr)" --no-school-docs >> "%LOG%" 2>&1
echo === S2 done %DATE% %TIME% === >> "%LOG%"
echo === S3 start %DATE% %TIME% === >> "%LOG%"
"%PY%" tools\fold_test.py --core results\cloud\20261016-0122\core_staged.pt --label "S3 120 steps (D34 lr)" --steps 120 >> "%LOG%" 2>&1
echo === S3 done %DATE% %TIME% === >> "%LOG%"
echo === ALL DONE === >> "%LOG%"
