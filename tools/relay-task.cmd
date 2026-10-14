@echo off
rem "Sapling Relay" scheduled task: the owner's door (127.0.0.1:7338 + phone relay). Log: results/relay.log
cd /d "%~dp0.."
"C:\Users\lilli\scoop\apps\python312\current\python.exe" tools\owner_relay.py >> results\relay.stdout.log 2>&1
