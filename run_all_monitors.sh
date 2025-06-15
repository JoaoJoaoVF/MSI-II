#!/bin/bash

# Script para executar todos os real_time_monitor.py com os arquivos CSV na pasta data
# Executa sequencialmente para todos os modelos: DistilBERT, MiniLM e TinyBERT
# Versão otimizada com controle de espaço em disco

echo "=== Iniciando execução de todos os monitores em tempo real ==="
echo "Data/Hora: $(date)"
echo ""

# Função para verificar espaço em disco
check_disk_space() {
    local required_gb=5  # Espaço mínimo necessário em GB
    local available_kb=$(df . | tail -1 | awk '{print $4}')
    local available_gb=$((available_kb / 1024 / 1024))
    
    echo "💾 Espaço disponível: ${available_gb}GB"
    
    if [ $available_gb -lt $required_gb ]; then
        echo "❌ ERRO: Espaço insuficiente! Necessário pelo menos ${required_gb}GB"
        echo "💡 Sugestões:"
        echo "   - Limpar arquivos temporários: rm -f */result-*.txt"
        echo "   - Compactar resultados antigos: tar -czf results_backup.tar.gz */result-*.txt"
        echo "   - Usar um diretório com mais espaço"
        return 1
    fi
    
    echo "✅ Espaço suficiente disponível"
    return 0
}

# Função para limpar arquivos temporários grandes
cleanup_temp_files() {
    local model_dir=$1
    echo "🧹 Limpando arquivos temporários em $model_dir..."
    
    # Remove arquivos de resultado muito grandes (>100MB) e vazios
    find "$model_dir" -name "result-*.txt" -size +100M -delete 2>/dev/null || true
    find "$model_dir" -name "result-*.txt" -size 0 -delete 2>/dev/null || true
    
    # Compacta logs de ataque se existirem
    if [ -f "$model_dir/attack_log.json" ] && [ $(stat -f%z "$model_dir/attack_log.json" 2>/dev/null || stat -c%s "$model_dir/attack_log.json" 2>/dev/null || echo 0) -gt 10485760 ]; then
        gzip "$model_dir/attack_log.json" 2>/dev/null || true
    fi
}

# Função para extrair métricas detalhadas de um arquivo de resultado
extract_and_log_metrics() {
    local result_file=$1
    local csv_name=$2
    local processing_time=$3
    local status=$4
    
    if [ ! -f "$result_file" ] || [ ! -s "$result_file" ]; then
        echo "$csv_name,$status,$processing_time,0,0,0,0,0,0,0"
        return
    fi
    
    # Extrair métricas do arquivo de resultado
    local total_predictions=$(grep -c "Predição:" "$result_file" 2>/dev/null || echo "0")
    local correct_predictions=$(grep -c "✅" "$result_file" 2>/dev/null || echo "0")
    local incorrect_predictions=$((total_predictions - correct_predictions))
    local file_size=$(stat -f%z "$result_file" 2>/dev/null || stat -c%s "$result_file" 2>/dev/null || echo 0)
    
    # Calcular métricas básicas
    local accuracy="0"
    if [ $total_predictions -gt 0 ]; then
        accuracy=$(echo "scale=4; $correct_predictions / $total_predictions" | bc 2>/dev/null || echo "0")
    fi
    
    # Extrair métricas do final do arquivo (se disponíveis)
    local precision=$(grep "Precision:" "$result_file" | tail -1 | awk '{print $2}' 2>/dev/null || echo "0")
    local recall=$(grep "Recall:" "$result_file" | tail -1 | awk '{print $2}' 2>/dev/null || echo "0")
    local f1_score=$(grep "F1-Score:" "$result_file" | tail -1 | awk '{print $2}' 2>/dev/null || echo "0")
    
    echo "$csv_name,$status,$processing_time,$correct_predictions,$incorrect_predictions,$accuracy,$precision,$recall,$f1_score,$file_size"
}

# Função para extrair métricas de arquivo já existente
extract_metrics_from_result() {
    local result_file=$1
    local csv_name=$2
    
    extract_and_log_metrics "$result_file" "$csv_name" "N/A" "JÁ_PROCESSADO"
}

# Função para verificar e registrar falhas de predição
check_prediction_failures() {
    local result_file=$1
    local csv_name=$2
    
    if [ ! -f "$result_file" ] || [ ! -s "$result_file" ]; then
        return
    fi
    
    # Procurar por linhas com predições incorretas (marcadas com ❌)
    local failures=$(grep -n "❌" "$result_file" 2>/dev/null || true)
    
    if [ -n "$failures" ]; then
        echo ""
        echo "=== FALHAS DE PREDIÇÃO EM $csv_name ==="
        echo "Data: $(date)"
        echo "$failures"
        echo ""
        
        # Contar tipos de falhas
        local ddos_failures=$(echo "$failures" | grep -i "ddos" | wc -l)
        local malware_failures=$(echo "$failures" | grep -i "malware" | wc -l)
        local benign_failures=$(echo "$failures" | grep -i "benign" | wc -l)
        
        echo "Resumo de falhas em $csv_name:"
        echo "  - DDoS: $ddos_failures"
        echo "  - Malware: $malware_failures"
        echo "  - Tráfego Benigno: $benign_failures"
        echo "  - Total: $(echo "$failures" | wc -l)"
        echo ""
    fi
}

# Definir caminhos
DATA_DIR="./data"
DISTILBERT_DIR="./DistilBERT"
MINILM_DIR="./MiniLM"
TINYBERT_DIR="./TinyBERT"
RESULTS_DIR="./analysis_results"

# Criar diretório de resultados se não existir
mkdir -p "$RESULTS_DIR"/{DistilBERT,MiniLM,TinyBERT}

# Verificar espaço em disco antes de iniciar
if ! check_disk_space; then
    exit 1
fi

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
    local BATCH_SIZE=5  # Reduzido para economizar espaço mas garantir completude
    
    echo "🚀 Iniciando processamento COMPLETO para modelo: $MODEL_NAME"
    echo "📁 Diretório: $MODEL_DIR"
    echo "🎯 OBJETIVO: Processar TODOS os arquivos CSV com métricas completas"
    
    # Verificar se o script existe
    if [ ! -f "$MODEL_DIR/$MONITOR_SCRIPT" ]; then
        echo "❌ Arquivo $MONITOR_SCRIPT não encontrado em $MODEL_DIR"
        return 1
    fi
    
    # Limpar apenas arquivos vazios, manter os válidos
    echo "🧹 Limpando apenas arquivos vazios (mantendo dados válidos)..."
    find "$MODEL_DIR" -name "result-*.txt" -size 0 -delete 2>/dev/null || true
    
    # Mudar para o diretório do modelo
    cd "$MODEL_DIR" || exit 1
    
    # Contador de arquivos processados
    local processed=0
    local success_count=0
    local error_count=0
    local skipped_count=0
    local total_files=$(find "../$DATA_DIR" -name "*.csv" | wc -l)
    local results_summary="../$RESULTS_DIR/${MODEL_NAME}/processing_summary.txt"
    local metrics_summary="../$RESULTS_DIR/${MODEL_NAME}/metrics_summary.txt"
    local failed_predictions="../$RESULTS_DIR/${MODEL_NAME}/failed_predictions.txt"
    
    echo "📈 Processando $total_files arquivos CSV - MODO COMPLETO"
    echo "📊 Métricas serão coletadas para TODOS os arquivos"
    echo "📄 Resumo: $results_summary"
    echo "📊 Métricas: $metrics_summary"
    echo "⚠️  Falhas de predição: $failed_predictions"
    
    # Inicializar arquivos de resumo
    {
        echo "=== RESUMO DE PROCESSAMENTO COMPLETO - $MODEL_NAME ==="
        echo "Data/Hora de início: $(date)"
        echo "Total de arquivos: $total_files"
        echo "Objetivo: Processar TODOS os arquivos com métricas completas"
        echo ""
    } > "$results_summary"
    
    {
        echo "=== MÉTRICAS DETALHADAS - $MODEL_NAME ==="
        echo "Data/Hora: $(date)"
        echo "Arquivo,Status,Tempo_Processamento,Predições_Corretas,Predições_Incorretas,Accuracy,Precision,Recall,F1_Score,Tamanho_Resultado"
        echo ""
    } > "$metrics_summary"
    
    {
        echo "=== FALHAS DE PREDIÇÃO - $MODEL_NAME ==="
        echo "Data/Hora: $(date)"
        echo "Registro de casos onde o modelo não predisse corretamente"
        echo ""
    } > "$failed_predictions"
    
    # Processar cada arquivo CSV - GARANTIR COMPLETUDE
    local batch_count=0
    echo "🔄 Iniciando processamento sequencial de TODOS os arquivos..."
    
    for csv_file in "../$DATA_DIR"/*.csv; do
        if [ -f "$csv_file" ]; then
            csv_basename=$(basename "$csv_file")
            processed=$((processed + 1))
            batch_count=$((batch_count + 1))
            
            echo "  [$processed/$total_files] 🎯 PROCESSANDO: $csv_basename"
            
            # Nome do arquivo de resultado
            local result_file="result-$(echo $MODEL_NAME | tr '[:upper:]' '[:lower:]')-part-$csv_basename.txt"
            
            # Verificar se já foi processado com sucesso (arquivo existe e não está vazio)
            if [ -f "$result_file" ] && [ -s "$result_file" ]; then
                local file_size=$(stat -f%z "$result_file" 2>/dev/null || stat -c%s "$result_file" 2>/dev/null || echo 0)
                if [ $file_size -gt 1000 ]; then  # Pelo menos 1KB de dados
                    echo "    ✅ JÁ PROCESSADO: $csv_basename (${file_size} bytes) - MANTENDO"
                    success_count=$((success_count + 1))
                    skipped_count=$((skipped_count + 1))
                    echo "JÁ_PROCESSADO: $csv_basename - $(date) - ${file_size} bytes" >> "$results_summary"
                    
                    # Extrair métricas do arquivo existente (se possível)
                    echo "    📊 Extraindo métricas do arquivo existente..."
                    extract_metrics_from_result "$result_file" "$csv_basename" >> "$metrics_summary"
                    
                    echo "    ---"
                    continue
                fi
            fi
            
            # Verificar espaço em disco a cada lote
            if [ $((batch_count % BATCH_SIZE)) -eq 0 ]; then
                local available_kb=$(df . | tail -1 | awk '{print $4}')
                local available_mb=$((available_kb / 1024))
                
                echo "    💾 Verificando espaço: ${available_mb}MB disponível"
                
                if [ $available_mb -lt 300 ]; then  # Menos de 300MB
                    echo "    🧹 Espaço baixo, limpando apenas arquivos vazios..."
                    find . -name "result-*.txt" -size 0 -delete 2>/dev/null || true
                    
                    # Verificar novamente
                    available_kb=$(df . | tail -1 | awk '{print $4}')
                    available_mb=$((available_kb / 1024))
                    
                    if [ $available_mb -lt 100 ]; then  # Menos de 100MB
                        echo "    ⚠️  ESPAÇO CRÍTICO (${available_mb}MB) - Continuando com arquivo menor"
                        echo "    📝 Forçando processamento com saída reduzida..."
                    fi
                fi
            fi
            
            # Registro do início do processamento
            local start_time=$(date +%s)
            echo "    ⏱️  Iniciando processamento às $(date)"
            
            # Executar o monitor com timeout estendido e captura detalhada
            echo "    🚀 Executando análise do modelo $MODEL_NAME..."
            
            # Executar com timeout de 10 minutos por arquivo (estendido para garantir completude)
            timeout 600 python "$MONITOR_SCRIPT" --simulate "$csv_file" --delay 0.001 2>&1 | tee "${csv_basename}_execution.log"
            local exit_code=$?
            local end_time=$(date +%s)
            local processing_time=$((end_time - start_time))
            
            if [ $exit_code -eq 0 ]; then
                success_count=$((success_count + 1))
                echo "    ✅ SUCESSO: $csv_basename (${processing_time}s)"
                
                # Verificar se o arquivo de resultado foi criado e analisar métricas
                if [ -f "$result_file" ] && [ -s "$result_file" ]; then
                    local file_size=$(stat -f%z "$result_file" 2>/dev/null || stat -c%s "$result_file" 2>/dev/null || echo 0)
                    echo "      📊 Resultado: $result_file (${file_size} bytes)"
                    
                    # Extrair e registrar métricas
                    echo "      📈 Extraindo métricas detalhadas..."
                    extract_and_log_metrics "$result_file" "$csv_basename" "$processing_time" "SUCESSO" >> "$metrics_summary"
                    
                    # Verificar falhas de predição
                    echo "      🔍 Verificando falhas de predição..."
                    check_prediction_failures "$result_file" "$csv_basename" >> "$failed_predictions"
                    
                    echo "SUCESSO: $csv_basename - ${processing_time}s - ${file_size} bytes - $(date)" >> "$results_summary"
                else
                    echo "      ⚠️  AVISO: Arquivo de resultado vazio ou não criado"
                    echo "AVISO: $csv_basename - Resultado vazio após ${processing_time}s - $(date)" >> "$results_summary"
                    echo "$csv_basename,VAZIO,${processing_time},0,0,0,0,0,0,0" >> "$metrics_summary"
                fi
                
            elif [ $exit_code -eq 124 ]; then
                error_count=$((error_count + 1))
                echo "    ⏰ TIMEOUT: $csv_basename (>10min)"
                echo "TIMEOUT: $csv_basename - >10min - $(date)" >> "$results_summary"
                echo "$csv_basename,TIMEOUT,600,0,0,0,0,0,0,0" >> "$metrics_summary"
                
            else
                error_count=$((error_count + 1))
                echo "    ❌ ERRO: $csv_basename (código: $exit_code, ${processing_time}s)"
                
                # Tentar extrair informações do log de erro
                if [ -f "${csv_basename}_execution.log" ]; then
                    echo "      📋 Log de erro disponível, tentando diagnóstico..."
                    local error_info=$(tail -5 "${csv_basename}_execution.log" | grep -E "(Error|Exception|OSError)" | head -1)
                    echo "      🔍 Erro: $error_info"
                fi
                
                echo "ERRO: $csv_basename - Código $exit_code - ${processing_time}s - $(date)" >> "$results_summary"
                echo "$csv_basename,ERRO,${processing_time},0,0,0,0,0,0,0" >> "$metrics_summary"
                
                # Para erros de espaço, tentar limpeza e continuar
                if [ $exit_code -eq 1 ] || grep -q "No space left" "${csv_basename}_execution.log" 2>/dev/null; then
                    echo "      🧹 Erro de espaço detectado, limpando e continuando..."
                    find . -name "result-*.txt" -size 0 -delete 2>/dev/null || true
                    find . -name "*_execution.log" -mmin +5 -delete 2>/dev/null || true
                fi
            fi
            
            # Limpeza do log de execução para economizar espaço
            rm -f "${csv_basename}_execution.log" 2>/dev/null || true
            
            echo "    ---"
        fi
    done
    
    # Finalizar resumo com estatísticas completas
    {
        echo ""
        echo "=== ESTATÍSTICAS FINAIS COMPLETAS ==="
        echo "Total processados: $processed/$total_files"
        echo "Sucessos: $success_count"
        echo "Erros: $error_count"
        echo "Já processados (reutilizados): $skipped_count"
        echo "Taxa de sucesso: $(( (success_count * 100) / processed ))%"
        echo "Taxa de completude: $(( ((success_count + skipped_count) * 100) / total_files ))%"
        echo ""
        echo "=== GARANTIA DE COMPLETUDE ==="
        echo "Arquivos com métricas coletadas: $((success_count + skipped_count))"
        echo "Arquivos restantes: $((total_files - success_count - skipped_count))"
        echo "Data/Hora de conclusão: $(date)"
    } >> "$results_summary"
    
    echo "✅ $MODEL_NAME: Processamento COMPLETO finalizado"
    echo "   📊 Sucessos: $success_count/$processed"
    echo "   🔄 Reutilizados: $skipped_count/$processed"
    echo "   ❌ Erros: $error_count/$processed"
    echo "   🎯 Completude: $(( ((success_count + skipped_count) * 100) / total_files ))%"
    echo "   📄 Resumo: $results_summary"
    echo "   📊 Métricas: $metrics_summary"
    echo "   ⚠️  Falhas: $failed_predictions"
    echo ""
    
    # Voltar ao diretório raiz
    cd .. || exit 1
    
    return 0
}

# Função para processar todos os modelos em paralelo (opcional)
run_all_models_parallel() {
    echo "🔄 Executando todos os modelos em PARALELO..."
    echo "⚠️  Nota: Isso pode consumir muita CPU, memória e espaço em disco"
    echo ""
    
    # Verificar espaço em disco antes de iniciar paralelo
    local available_kb=$(df . | tail -1 | awk '{print $4}')
    local available_gb=$((available_kb / 1024 / 1024))
    
    if [ $available_gb -lt 10 ]; then
        echo "❌ Espaço insuficiente para execução paralela (${available_gb}GB < 10GB)"
        echo "💡 Use execução sequencial ou libere mais espaço"
        return 1
    fi
    
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
    echo "🔄 Executando TODOS OS MODELOS SEQUENCIALMENTE..."
    echo "🎯 OBJETIVO: Coletar métricas completas de TODOS os 3 modelos para TODOS os arquivos CSV"
    echo "📊 Garantindo que nenhum arquivo ou modelo seja pulado"
    echo ""
    
    # Verificar espaço antes de cada modelo
    local start_time=$(date +%s)
    
    echo "📋 === PLANO DE EXECUÇÃO COMPLETA ==="
    echo "1. DistilBERT: Todos os $(find $DATA_DIR -name "*.csv" | wc -l) arquivos CSV"
    echo "2. MiniLM: Todos os $(find $DATA_DIR -name "*.csv" | wc -l) arquivos CSV"
    echo "3. TinyBERT: Todos os $(find $DATA_DIR -name "*.csv" | wc -l) arquivos CSV"
    echo "Total de processamentos: $(($(find $DATA_DIR -name "*.csv" | wc -l) * 3))"
    echo ""
    
    # Criar arquivo de controle geral
    local general_summary="$RESULTS_DIR/complete_execution_summary.txt"
    {
        echo "=== EXECUÇÃO COMPLETA DE TODOS OS MODELOS ==="
        echo "Data/Hora de início: $(date)"
        echo "Total de arquivos CSV: $(find $DATA_DIR -name "*.csv" | wc -l)"
        echo "Modelos: DistilBERT, MiniLM, TinyBERT"
        echo "Objetivo: Métricas completas para todos os modelos e arquivos"
        echo ""
    } > "$general_summary"
    
    # Executar DistilBERT
    echo "🟦 === FASE 1/3: DISTILBERT ==="
    check_disk_space || echo "⚠️  Continuando mesmo com pouco espaço..."
    run_model_monitor "DistilBERT" "$DISTILBERT_DIR"
    local distilbert_status=$?
    echo "DISTILBERT_STATUS: $distilbert_status - $(date)" >> "$general_summary"
    echo ""
    
    # Executar MiniLM
    echo "🟩 === FASE 2/3: MINILM ==="
    check_disk_space || echo "⚠️  Continuando mesmo com pouco espaço..."
    run_model_monitor "MiniLM" "$MINILM_DIR"
    local minilm_status=$?
    echo "MINILM_STATUS: $minilm_status - $(date)" >> "$general_summary"
    echo ""
    
    # Executar TinyBERT
    echo "🟨 === FASE 3/3: TINYBERT ==="
    check_disk_space || echo "⚠️  Continuando mesmo com pouco espaço..."
    run_model_monitor "TinyBERT" "$TINYBERT_DIR"
    local tinybert_status=$?
    echo "TINYBERT_STATUS: $tinybert_status - $(date)" >> "$general_summary"
    echo ""
    
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    
    # Finalizar resumo geral
    {
        echo ""
        echo "=== RESUMO FINAL DA EXECUÇÃO COMPLETA ==="
        echo "Tempo total: ${total_time}s ($((total_time / 60))min)"
        echo "Status DistilBERT: $distilbert_status"
        echo "Status MiniLM: $minilm_status"
        echo "Status TinyBERT: $tinybert_status"
        echo "Data/Hora de conclusão: $(date)"
        echo ""
        echo "=== VERIFICAÇÃO DE COMPLETUDE ==="
    } >> "$general_summary"
    
    # Verificar completude para cada modelo
    for model in DistilBERT MiniLM TinyBERT; do
        local model_dir_path=""
        case $model in
            "DistilBERT") model_dir_path="$DISTILBERT_DIR" ;;
            "MiniLM") model_dir_path="$MINILM_DIR" ;;
            "TinyBERT") model_dir_path="$TINYBERT_DIR" ;;
        esac
        
        if [ -d "$model_dir_path" ]; then
            local result_count=$(find "$model_dir_path" -name "result-*.txt" -size +0c | wc -l)
            local csv_count=$(find "$DATA_DIR" -name "*.csv" | wc -l)
            local completeness=$((result_count * 100 / csv_count))
            
            echo "$model: $result_count/$csv_count arquivos processados (${completeness}%)" >> "$general_summary"
            
            if [ $completeness -lt 100 ]; then
                echo "  ⚠️  ATENÇÃO: $model não processou todos os arquivos!" >> "$general_summary"
                # Listar arquivos faltantes
                echo "  Arquivos faltantes:" >> "$general_summary"
                for csv_file in "$DATA_DIR"/*.csv; do
                    local csv_basename=$(basename "$csv_file")
                    local expected_result="result-$(echo $model | tr '[:upper:]' '[:lower:]')-part-$csv_basename.txt"
                    if [ ! -f "$model_dir_path/$expected_result" ] || [ ! -s "$model_dir_path/$expected_result" ]; then
                        echo "    - $csv_basename" >> "$general_summary"
                    fi
                done
            else
                echo "  ✅ $model: TODOS os arquivos processados!" >> "$general_summary"
            fi
        fi
    done
    
    echo ""
    echo "🎉 === EXECUÇÃO COMPLETA FINALIZADA ==="
    echo "📄 Resumo geral: $general_summary"
    echo ""
    
    # Mostrar estatísticas consolidadas
    show_consolidated_stats
}

# Função para mostrar estatísticas consolidadas
show_consolidated_stats() {
    echo "📊 === ESTATÍSTICAS CONSOLIDADAS DE TODOS OS MODELOS ==="
    echo ""
    
    local csv_total=$(find "$DATA_DIR" -name "*.csv" | wc -l)
    echo "📁 Total de arquivos CSV: $csv_total"
    echo ""
    
    for model in DistilBERT MiniLM TinyBERT; do
        echo "🔸 === $model ==="
        
        local model_dir_path=""
        case $model in
            "DistilBERT") model_dir_path="$DISTILBERT_DIR" ;;
            "MiniLM") model_dir_path="$MINILM_DIR" ;;
            "TinyBERT") model_dir_path="$TINYBERT_DIR" ;;
        esac
        
        if [ -d "$model_dir_path" ]; then
            # Contar resultados
            local result_files=$(find "$model_dir_path" -name "result-*.txt" -size +0c | wc -l)
            local empty_files=$(find "$model_dir_path" -name "result-*.txt" -size 0 | wc -l)
            local completeness=$((result_files * 100 / csv_total))
            
            echo "  📊 Arquivos processados: $result_files/$csv_total (${completeness}%)"
            echo "  📄 Arquivos válidos: $result_files"
            echo "  ⚠️  Arquivos vazios: $empty_files"
            
            # Verificar arquivos de métricas
            local metrics_file="$RESULTS_DIR/$model/metrics_summary.txt"
            if [ -f "$metrics_file" ]; then
                local metrics_lines=$(grep -c "," "$metrics_file" 2>/dev/null || echo "0")
                echo "  📈 Linhas de métricas: $metrics_lines"
                
                # Estatísticas de sucesso do arquivo de métricas
                local success_count=$(grep -c ",SUCESSO," "$metrics_file" 2>/dev/null || echo "0")
                local error_count=$(grep -c ",ERRO," "$metrics_file" 2>/dev/null || echo "0")
                local timeout_count=$(grep -c ",TIMEOUT," "$metrics_file" 2>/dev/null || echo "0")
                
                echo "  ✅ Sucessos: $success_count"
                echo "  ❌ Erros: $error_count"
                echo "  ⏰ Timeouts: $timeout_count"
            fi
            
            # Verificar arquivo de falhas
            local failures_file="$RESULTS_DIR/$model/failed_predictions.txt"
            if [ -f "$failures_file" ]; then
                local failure_sections=$(grep -c "=== FALHAS DE PREDIÇÃO" "$failures_file" 2>/dev/null || echo "0")
                echo "  🚨 Arquivos com falhas de predição: $failure_sections"
            fi
            
            # Tamanho total dos resultados
            if command -v du >/dev/null 2>&1; then
                local model_size=$(du -sh "$model_dir_path" 2>/dev/null | cut -f1)
                echo "  💾 Tamanho total: $model_size"
            fi
            
        else
            echo "  ❌ Diretório não encontrado: $model_dir_path"
        fi
        echo ""
    done
    
    # Criar arquivo de estatísticas consolidadas
    local consolidated_stats="$RESULTS_DIR/consolidated_statistics.txt"
    {
        echo "=== ESTATÍSTICAS CONSOLIDADAS ==="
        echo "Gerado em: $(date)"
        echo ""
        echo "Total de arquivos CSV: $csv_total"
        echo ""
        
        for model in DistilBERT MiniLM TinyBERT; do
            echo "=== $model ==="
            case $model in
                "DistilBERT") model_dir_path="$DISTILBERT_DIR" ;;
                "MiniLM") model_dir_path="$MINILM_DIR" ;;
                "TinyBERT") model_dir_path="$TINYBERT_DIR" ;;
            esac
            
            if [ -d "$model_dir_path" ]; then
                local result_files=$(find "$model_dir_path" -name "result-*.txt" -size +0c | wc -l)
                local completeness=$((result_files * 100 / csv_total))
                echo "Completude: ${completeness}% ($result_files/$csv_total)"
                
                local metrics_file="$RESULTS_DIR/$model/metrics_summary.txt"
                if [ -f "$metrics_file" ]; then
                    local success_count=$(grep -c ",SUCESSO," "$metrics_file" 2>/dev/null || echo "0")
                    local error_count=$(grep -c ",ERRO," "$metrics_file" 2>/dev/null || echo "0")
                    echo "Sucessos: $success_count"
                    echo "Erros: $error_count"
                fi
            else
                echo "Status: Diretório não encontrado"
            fi
            echo ""
        done
        
    } > "$consolidated_stats"
    
    echo "📄 Estatísticas salvas em: $consolidated_stats"
}

# Função para verificar se algum modelo precisa ser reprocessado
check_missing_files() {
    echo "🔍 === VERIFICANDO ARQUIVOS FALTANTES ==="
    echo ""
    
    local csv_files=($(find "$DATA_DIR" -name "*.csv" -exec basename {} \;))
    local missing_any=false
    
    for model in DistilBERT MiniLM TinyBERT; do
        echo "🔸 Verificando $model..."
        
        local model_dir_path=""
        case $model in
            "DistilBERT") model_dir_path="$DISTILBERT_DIR" ;;
            "MiniLM") model_dir_path="$MINILM_DIR" ;;
            "TinyBERT") model_dir_path="$TINYBERT_DIR" ;;
        esac
        
        local missing_files=()
        for csv_file in "${csv_files[@]}"; do
            local expected_result="result-$(echo $model | tr '[:upper:]' '[:lower:]')-part-$csv_file.txt"
            if [ ! -f "$model_dir_path/$expected_result" ] || [ ! -s "$model_dir_path/$expected_result" ]; then
                missing_files+=("$csv_file")
                missing_any=true
            fi
        done
        
        if [ ${#missing_files[@]} -eq 0 ]; then
            echo "  ✅ Todos os arquivos processados"
        else
            echo "  ⚠️  Arquivos faltantes (${#missing_files[@]}/${#csv_files[@]}):"
            for missing in "${missing_files[@]}"; do
                echo "    - $missing"
            done
        fi
        echo ""
    done
    
    if [ "$missing_any" = true ]; then
        echo "⚠️  ATENÇÃO: Alguns arquivos não foram processados em todos os modelos"
        echo "💡 Recomendação: Execute novamente os modelos com arquivos faltantes"
        return 1
    else
        echo "✅ PERFEITO: Todos os arquivos foram processados em todos os modelos!"
        return 0
    fi
}
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
    echo "4. ⭐ EXECUTAR TODOS OS MODELOS COMPLETO (recomendado)"
    echo "5. Executar todos em paralelo (alto uso de recursos)"
    echo "6. Verificar arquivos faltantes"
    echo "7. Mostrar estatísticas consolidadas"
    echo "8. Limpar arquivos temporários e resultados vazios"
    echo "9. Mostrar relatório de espaço em disco"
    echo "10. Sair"
    echo ""
    echo "💡 DICA: Opção 4 é recomendada para coletar TODAS as métricas"
    echo ""
}

# Função para limpeza geral
cleanup_all() {
    echo "🧹 Limpando arquivos temporários e resultados vazios..."
    
    for dir in "$DISTILBERT_DIR" "$MINILM_DIR" "$TINYBERT_DIR"; do
        if [ -d "$dir" ]; then
            echo "  Limpando $dir..."
            cleanup_temp_files "$dir"
        fi
    done
    
    echo "✅ Limpeza concluída"
}

# Função para mostrar relatório de espaço
show_disk_report() {
    echo "💾 === RELATÓRIO DE ESPAÇO EM DISCO ==="
    df -h .
    echo ""
    
    echo "📁 Tamanho dos diretórios:"
    for dir in "$DISTILBERT_DIR" "$MINILM_DIR" "$TINYBERT_DIR" "$RESULTS_DIR"; do
        if [ -d "$dir" ]; then
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo "  $dir: $size"
        fi
    done
    echo ""
    
    echo "📄 Arquivos de resultado:"
    find . -name "result-*.txt" -exec ls -lh {} \; | awk '{print "  " $9 ": " $5}' | head -10
    if [ $(find . -name "result-*.txt" | wc -l) -gt 10 ]; then
        echo "  ... e mais $(( $(find . -name "result-*.txt" | wc -l) - 10 )) arquivos"
    fi
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
                echo "⚠️  Tem certeza? Isso pode usar muita CPU/memória/disco (y/n):"
                read -p "" confirm
                if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                    run_all_models_parallel
                fi
                break
                ;;
            6)
                check_missing_files
                ;;
            7)
                show_consolidated_stats
                ;;
            8)
                cleanup_all
                ;;
            9)
                show_disk_report
                ;;
            10)
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
        "--cleanup"|"-c")
            cleanup_all
            ;;
        "--disk"|"-s")
            show_disk_report
            ;;
        "--check"|"-ck")
            check_missing_files
            ;;
        "--stats"|"-st")
            show_consolidated_stats
            ;;
        "--missing"|"-m")
            echo "🔍 Verificando arquivos faltantes..."
            check_missing_files
            ;;
        "--complete"|"-comp")
            echo "🎯 Executando processamento COMPLETO de todos os modelos..."
            run_all_models_sequential
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
            echo "  --cleanup     Limpar arquivos temporários"
            echo "  --disk        Mostrar relatório de espaço em disco"
            echo "  --check       Verificar arquivos faltantes"
            echo "  --stats       Mostrar estatísticas consolidadas"
            echo "  --missing     Verificar arquivos faltantes"
            echo "  --complete    Executar processamento completo"
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

# Mostrar estatísticas finais
echo "📊 === ESTATÍSTICAS FINAIS ==="
for model in DistilBERT MiniLM TinyBERT; do
    summary_file="$RESULTS_DIR/$model/processing_summary.txt"
    if [ -f "$summary_file" ]; then
        echo "🔸 $model:"
        grep -E "(Sucessos:|Erros:|Taxa de sucesso:)" "$summary_file" | sed 's/^/    /'
    fi
done
echo ""

# Mostrar arquivos de resultado gerados
echo "📄 Resumos de processamento:"
find "$RESULTS_DIR" -name "processing_summary.txt" -type f | sort

echo ""
echo "📄 Arquivos de resultado recentes (últimos 10):"
find . -name "result-*.txt" -type f -mmin -60 -size +0c | sort | tail -10

echo ""
echo "� Espaço final em disco:"
df -h . | tail -1

echo ""
echo "�💡 Dicas:"
echo "   - Verifique os resumos em $RESULTS_DIR para análise detalhada"
echo "   - Use '$0 --cleanup' para limpar arquivos temporários"
echo "   - Use '$0 --disk' para ver relatório completo de espaço"
echo "   - Os logs de ataque estão em attack_log.json em cada pasta de modelo"
