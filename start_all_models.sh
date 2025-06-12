#!/bin/bash
cd "$(dirname "$0")"

echo "🚀 Iniciando todos os modelos..."

# Função para iniciar modelo em background
start_model() {
    local model=$1
    echo "Iniciando $model..."
    source venv/bin/activate
    cd "$model"
    python3 realtime_network_monitor.py --interactive &
    echo "$!" > "../${model,,}.pid"
    cd ..
}

# Iniciar cada modelo
start_model "DistilBERT"
start_model "TinyBERT"
start_model "MiniLM"

echo "✅ Todos os modelos iniciados!"
echo "PIDs salvos em: distilbert.pid, tinybert.pid, minilm.pid"
echo "Para parar todos: ./stop_all_models.sh"
