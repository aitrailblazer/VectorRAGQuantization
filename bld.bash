#!/usr/bin/env bash

#!/usr/bin/env bash
set -e  # Exit on any error


echo "Running main.py (all local + global workflows)..."
python main.py > 1.log 2>&1


