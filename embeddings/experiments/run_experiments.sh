#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

conda run -n FinGPT --no-capture-output python msrr.py
echo "MSRR done."

conda run -n FinGPT --no-capture-output python mse.py
echo "MSE done."

echo "All experiments complete."
