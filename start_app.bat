@echo off
REM Start the Smart Sensor Monitoring web app (Windows). First run installs the Python packages.
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
start "" http://localhost:8050
python -m hsm.cli app --data appdata --port 8050
