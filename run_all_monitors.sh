#!/bin/bash

# Script para executar todos os real_time_monitor.py com os arquivos CSV na pasta data
# Executa sequencialmente para todos os modelos: DistilBERT, MiniLM e TinyBERT

echo "=== Iniciando execução de todos os monitores em tempo real ==="
echo "Data/Hora: $(date)"
echo ""

# Definir caminhos
DATA_DIR="./data"
DISTILBERT_DIR="./DistilBERT"
MINILM_DIR="./MiniLM"
TINYBERT_DIR="./TinyBERT"

# Verificar se os diretórios existem
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Erro: Diretório $DATA_DIR não encontrado!"
    exit 1
fi

if [ ! -d "$DISTILBERT_DIR" ]; then
    echo "❌ Erro: Diretório $DISTILBERT_DIR não encontrado!"
    exit 1
fi

if [ ! -d "$MINILM_DIR" ]; then
    echo "❌ Erro: Diretório $MINILM_DIR não encontrado!"
    exit 1
fi

if [ ! -d "$TINYBERT_DIR" ]; then
    echo "❌ Erro: Diretório $TINYBERT_DIR não encontrado!"
    exit 1
fi

# Contar arquivos CSV
CSV_COUNT=$(find "$DATA_DIR" -name "*.csv" | wc -l)
echo "📊 Encontrados $CSV_COUNT arquivos CSV na pasta data"
echo ""

# Função para executar monitor de um modelo específico
run_model_monitor() {
    local MODEL_NAME=$1
    local MODEL_DIR=$2
    local MONITOR_SCRIPT="realtime_network_monitor.py"
    
    echo "🚀 Iniciando processamento para modelo: $MODEL_NAME"
    echo "📁 Diretório: $MODEL_DIR"
    
    # Verificar se o script existe
    if [ ! -f "$MODEL_DIR/$MONITOR_SCRIPT" ]; then
        echo "❌ Arquivo $MONITOR_SCRIPT não encontrado em $MODEL_DIR"
        return 1
    fi
    
    # Mudar para o diretório do modelo
    cd "$MODEL_DIR" || exit 1
    
    # Contador de arquivos processados
    local processed=0
    local total_files=$(find "../$DATA_DIR" -name "*.csv" | wc -l)
    
    echo "📈 Processando $total_files arquivos CSV..."
    
    # Processar cada arquivo CSV
    for csv_file in "../$DATA_DIR"/*.csv; do
        if [ -f "$csv_file" ]; then
            csv_basename=$(basename "$csv_file")
            processed=$((processed + 1))
            
            echo "  [$processed/$total_files] Processando: $csv_basename"
            
            # Executar o monitor com o arquivo CSV
            python "$MONITOR_SCRIPT" --simulate "$csv_file" --delay 0.001
            
            # Verificar se a execução foi bem-sucedida
            if [ $? -eq 0 ]; then
                echo "  ✅ Concluído: $csv_basename"
            else
                echo "  ❌ Erro ao processar: $csv_basename"
            fi
            
            echo "  ---"
        fi
    done
    
    echo "✅ $MODEL_NAME: Processamento concluído ($processed arquivos)"
    echo ""
    
    # Voltar ao diretório raiz
    cd .. || exit 1
}

# Função para processar todos os modelos em paralelo (opcional)
run_all_models_parallel() {
    echo "🔄 Executando todos os modelos em PARALELO..."
    echo "⚠️  Nota: Isso pode consumir muita CPU e memória"
    echo ""
    
    # Executar em background
    (run_model_monitor "DistilBERT" "$DISTILBERT_DIR") &
    DISTILBERT_PID=$!
    
    (run_model_monitor "MiniLM" "$MINILM_DIR") &
    MINILM_PID=$!
    
    (run_model_monitor "TinyBERT" "$TINYBERT_DIR") &
    TINYBERT_PID=$!
    
    # Aguardar conclusão de todos
    echo "⏳ Aguardando conclusão de todos os modelos..."
    wait $DISTILBERT_PID
    echo "✅ DistilBERT concluído"
    
    wait $MINILM_PID
    echo "✅ MiniLM concluído"
    
    wait $TINYBERT_PID
    echo "✅ TinyBERT concluído"
}

# Função para processar todos os modelos sequencialmente
run_all_models_sequential() {
    echo "🔄 Executando todos os modelos SEQUENCIALMENTE..."
    echo ""
    
    run_model_monitor "DistilBERT" "$DISTILBERT_DIR"
    run_model_monitor "MiniLM" "$MINILM_DIR" 
    run_model_monitor "TinyBERT" "$TINYBERT_DIR"
}

# Função para executar apenas um modelo específico
run_single_model() {
    local MODEL_CHOICE=$1
    
    case $MODEL_CHOICE in
        "distilbert"|"1")
            run_model_monitor "DistilBERT" "$DISTILBERT_DIR"
            ;;
        "minilm"|"2")
            run_model_monitor "MiniLM" "$MINILM_DIR"
            ;;
        "tinybert"|"3")
            run_model_monitor "TinyBERT" "$TINYBERT_DIR"
            ;;
        *)
            echo "❌ Modelo inválido: $MODEL_CHOICE"
            echo "Modelos disponíveis: distilbert, minilm, tinybert"
            exit 1
            ;;
    esac
}

# Menu de opções
show_menu() {
    echo "🎯 Escolha uma opção:"
    echo "1. Executar DistilBERT apenas"
    echo "2. Executar MiniLM apenas"
    echo "3. Executar TinyBERT apenas"
    echo "4. Executar todos sequencialmente (recomendado)"
    echo "5. Executar todos em paralelo (alto uso de recursos)"
    echo "6. Sair"
    echo ""
}

# Verificar argumentos da linha de comando
if [ $# -eq 0 ]; then
    # Modo interativo
    while true; do
        show_menu
        read -p "Digite sua escolha (1-6): " choice
        
        case $choice in
            1)
                run_single_model "distilbert"
                break
                ;;
            2)
                run_single_model "minilm"
                break
                ;;
            3)
                run_single_model "tinybert"
                break
                ;;
            4)
                run_all_models_sequential
                break
                ;;
            5)
                echo "⚠️  Tem certeza? Isso pode usar muita CPU/memória (y/n):"
                read -p "" confirm
                if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                    run_all_models_parallel
                fi
                break
                ;;
            6)
                echo "👋 Saindo..."
                exit 0
                ;;
            *)
                echo "❌ Opção inválida. Tente novamente."
                ;;
        esac
    done
else
    # Modo com argumentos
    case $1 in
        "--all"|"-a")
            run_all_models_sequential
            ;;
        "--parallel"|"-p")
            run_all_models_parallel
            ;;
        "--distilbert"|"-d")
            run_single_model "distilbert"
            ;;
        "--minilm"|"-m")
            run_single_model "minilm"
            ;;
        "--tinybert"|"-t")
            run_single_model "tinybert"
            ;;
        "--help"|"-h")
            echo "Uso: $0 [OPÇÃO]"
            echo ""
            echo "Opções:"
            echo "  --all         Executar todos os modelos sequencialmente"
            echo "  --parallel    Executar todos os modelos em paralelo"
            echo "  --distilbert  Executar apenas DistilBERT"
            echo "  --minilm      Executar apenas MiniLM"
            echo "  --tinybert    Executar apenas TinyBERT"
            echo "  --help        Mostrar esta ajuda"
            echo ""
            echo "Sem argumentos: modo interativo"
            ;;
        *)
            echo "❌ Argumento inválido: $1"
            echo "Use --help para ver as opções disponíveis"
            exit 1
            ;;
    esac
fi

# Relatório final
echo ""
echo "🎉 === EXECUÇÃO CONCLUÍDA ==="
echo "Data/Hora: $(date)"
echo ""

# Mostrar arquivos de resultado gerados
echo "📄 Arquivos de resultado gerados:"
find . -name "result-*-part-*.txt" -type f -mmin -60 | sort

echo ""
echo "💡 Dica: Verifique os arquivos de resultado para análise detalhada"
echo "💡 Os logs de ataque estão em attack_log.json em cada pasta de modelo"
