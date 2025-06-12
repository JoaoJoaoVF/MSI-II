#!/bin/bash
cd "$(dirname "$0")"

echo "🛑 Parando todos os modelos..."

for pid_file in distilbert.pid tinybert.pid minilm.pid; do
    if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Parando processo $pid..."
            kill "$pid"
            rm "$pid_file"
        else
            echo "Processo $pid já finalizado"
            rm "$pid_file"
        fi
    fi
done

echo "✅ Todos os modelos parados!"
