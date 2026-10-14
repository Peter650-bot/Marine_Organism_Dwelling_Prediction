#!/usr/bin/env bash
# Sequential (never concurrent) so wall-clock timings are comparable.
set -e
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
PYTHONPATH=~/venvs/maxent-site python3 extension/run_maxent.py --sweep
python3 extension/run_tabicl.py --sweep
PYTHONPATH=~/venvs/maxent-site python3 extension/run_maxent.py
python3 extension/run_tabicl.py
echo ALL_DONE
