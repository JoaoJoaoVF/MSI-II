#!/bin/bash
cd "$(dirname "$0")"

echo "🔥 Executando benchmark de todos os modelos..."

source venv/bin/activate

models=("DistilBERT" "TinyBERT" "MiniLM")

for model in "${models[@]}"; do
    echo ""
    echo "=== Benchmark $model ==="
    cd "$model"
    python3 realtime_network_monitor.py --benchmark
    cd ..
done

echo ""
echo "✅ Benchmark de todos os modelos concluído!"
