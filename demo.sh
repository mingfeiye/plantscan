#!/bin/bash
# The demo command. Runs everything on one scan and opens the report page.
#   ./demo.sh "scans/my scan.ply" --expected 9
#   ./demo.sh backup --expected 9
cd "$(dirname "$0")" || exit 1
exec venv/bin/python run_demo.py "$@"
