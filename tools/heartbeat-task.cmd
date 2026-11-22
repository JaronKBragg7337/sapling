@echo off
rem "Sapling Heartbeat" scheduled task: the process belongs to Task Scheduler, not a CLI session.
cd /d "%~dp0.."
"C:\Users\lilli\scoop\apps\python312\current\python.exe" -m sapling.heartbeat run >> results\heartbeat.log 2>&1
