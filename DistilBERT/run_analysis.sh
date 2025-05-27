#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

if [[ $# -eq 0 ]]; then
    echo "Uso: $0 <arquivo_csv_teste>"
    exit 1
fi

python3 performance_analyzer.py --test_data "$1"
