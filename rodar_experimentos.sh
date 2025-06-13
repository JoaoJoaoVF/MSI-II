#!/bin/bash
# run_all_simulations.sh
# Percorre todos os CSVs em /data e executa:
#  - minilm_network_monitor.py
#  - distilbert_network_monitor.py
#  - realtime_network_monitor.py (TinyBERT)

DATA_DIR="/data"

# Verifica se a pasta existe
if [ ! -d "$DATA_DIR" ]; then
  echo "Diretório de dados $DATA_DIR não encontrado."
  exit 1
fi

# Loop pelos CSVs
for csv in "$DATA_DIR"/*.csv; do
  echo "=== Processando $(basename "$csv") ==="

  echo -e "\n[MiniLM] Executando simulação..."
  python MiniLM/minilm_network_monitor.py --simulate "$csv"
  
  echo -e "\n[DistilBERT] Executando simulação..."
  python DistilBERT/distilbert_network_monitor.py --simulate "$csv"

  echo -e "\n[TinyBERT] Executando simulação..."
  python TinyBERT/realtime_network_monitor.py --simulate "$csv"

  echo -e "\n----------------------------------------\n"
done
