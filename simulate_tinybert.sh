#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
cd TinyBERT

if [[ $# -eq 0 ]]; then
    echo "Uso: $0 <arquivo_csv>"
    echo "Exemplo: $0 ../data/network_data.csv"
    echo "Os resultados serão salvos como: result-tinybert-part-<nome_csv>.txt"
    exit 1
fi

echo "Iniciando simulação TinyBERT com arquivo: $1"
echo "Resultado será salvo como: result-tinybert-part-$(basename $1 .csv).txt"

python3 realtime_network_monitor.py --simulate "$1" --delay 0.1
