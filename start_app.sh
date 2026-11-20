#!/usr/bin/env bash
# Start the Smart Sensor Monitoring web app (macOS / Linux). First run installs the Python packages.
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt
python3 -m hsm.cli app --data appdata --port 8050 &
sleep 4; (open http://localhost:8050 2>/dev/null || xdg-open http://localhost:8050 2>/dev/null) ; wait
