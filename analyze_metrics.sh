#!/bin/bash

# Script para análise detalhada das métricas coletadas
# Gera relatórios consolidados de todos os modelos

echo "=== ANALISADOR DE MÉTRICAS E RESULTADOS ==="
echo "Data/Hora: $(date)"
echo ""

# Diretórios
RESULTS_DIR="./analysis_results"
REPORTS_DIR="./analysis_reports"

# Criar diretório de relatórios
mkdir -p "$REPORTS_DIR"

# Função para analisar métricas de um modelo
analyze_model_metrics() {
    local model=$1
    local metrics_file="$RESULTS_DIR/$model/metrics_summary.txt"
    local failures_file="$RESULTS_DIR/$model/failed_predictions.txt"
    local report_file="$REPORTS_DIR/${model}_analysis_report.txt"
    
    echo "📊 Analisando métricas do $model..."
    
    if [ ! -f "$metrics_file" ]; then
        echo "❌ Arquivo de métricas não encontrado: $metrics_file"
        return 1
    fi
    
    {
        echo "=== RELATÓRIO DE ANÁLISE - $model ==="
        echo "Gerado em: $(date)"
        echo ""
        
        # Estatísticas básicas
        echo "=== ESTATÍSTICAS BÁSICAS ==="
        local total_lines=$(grep -c "," "$metrics_file" 2>/dev/null || echo "0")
        local header_lines=$(head -3 "$metrics_file" | grep -c "," || echo "1")
        local data_lines=$((total_lines - header_lines))
        
        echo "Total de registros: $data_lines"
        
        # Análise por status
        echo ""
        echo "=== ANÁLISE POR STATUS ==="
        local success_count=$(grep -c ",SUCESSO," "$metrics_file" 2>/dev/null || echo "0")
        local error_count=$(grep -c ",ERRO," "$metrics_file" 2>/dev/null || echo "0")
        local timeout_count=$(grep -c ",TIMEOUT," "$metrics_file" 2>/dev/null || echo "0")
        local already_processed=$(grep -c ",JÁ_PROCESSADO," "$metrics_file" 2>/dev/null || echo "0")
        
        echo "Sucessos: $success_count"
        echo "Erros: $error_count"
        echo "Timeouts: $timeout_count"
        echo "Já processados: $already_processed"
        
        if [ $data_lines -gt 0 ]; then
            echo "Taxa de sucesso: $(( (success_count * 100) / data_lines ))%"
            echo "Taxa de erro: $(( (error_count * 100) / data_lines ))%"
        fi
        
        # Análise de performance
        echo ""
        echo "=== ANÁLISE DE PERFORMANCE ==="
        
        # Tempos de processamento (excluindo N/A e TIMEOUT)
        grep ",SUCESSO," "$metrics_file" | cut -d',' -f3 | grep -v "N/A" | sort -n > /tmp/times_$model.txt
        if [ -s /tmp/times_$model.txt ]; then
            local min_time=$(head -1 /tmp/times_$model.txt)
            local max_time=$(tail -1 /tmp/times_$model.txt)
            local avg_time=$(awk '{sum+=$1} END {print sum/NR}' /tmp/times_$model.txt)
            
            echo "Tempo mínimo: ${min_time}s"
            echo "Tempo máximo: ${max_time}s"
            echo "Tempo médio: ${avg_time}s"
        fi
        rm -f /tmp/times_$model.txt
        
        # Análise de predições
        echo ""
        echo "=== ANÁLISE DE PREDIÇÕES ==="
        
        # Somar predições corretas e incorretas
        local total_correct=$(grep ",SUCESSO," "$metrics_file" | cut -d',' -f4 | awk '{sum+=$1} END {print sum+0}')
        local total_incorrect=$(grep ",SUCESSO," "$metrics_file" | cut -d',' -f5 | awk '{sum+=$1} END {print sum+0}')
        local total_predictions=$((total_correct + total_incorrect))
        
        echo "Total de predições: $total_predictions"
        echo "Predições corretas: $total_correct"
        echo "Predições incorretas: $total_incorrect"
        
        if [ $total_predictions -gt 0 ]; then
            local accuracy=$(echo "scale=4; $total_correct / $total_predictions * 100" | bc 2>/dev/null || echo "0")
            echo "Acurácia geral: ${accuracy}%"
        fi
        
        # Top 10 arquivos com mais predições incorretas
        echo ""
        echo "=== TOP 10 ARQUIVOS COM MAIS PREDIÇÕES INCORRETAS ==="
        grep ",SUCESSO," "$metrics_file" | sort -t',' -k5 -nr | head -10 | while IFS=',' read -r file status time correct incorrect accuracy precision recall f1 size; do
            echo "$file: $incorrect predições incorretas"
        done
        
        # Análise de falhas de predição (se disponível)
        if [ -f "$failures_file" ]; then
            echo ""
            echo "=== ANÁLISE DE FALHAS DE PREDIÇÃO ==="
            local files_with_failures=$(grep -c "=== FALHAS DE PREDIÇÃO" "$failures_file" 2>/dev/null || echo "0")
            echo "Arquivos com falhas de predição: $files_with_failures"
            
            # Tipos de falhas mais comuns
            echo ""
            echo "=== TIPOS DE ATAQUES COM MAIS FALHAS ==="
            grep -i "ddos\|dos" "$failures_file" | wc -l | xargs echo "DDoS/DoS:"
            grep -i "malware\|backdoor" "$failures_file" | wc -l | xargs echo "Malware:"
            grep -i "benign" "$failures_file" | wc -l | xargs echo "Tráfego Benigno:"
            grep -i "recon\|scan" "$failures_file" | wc -l | xargs echo "Reconhecimento:"
            grep -i "injection\|upload" "$failures_file" | wc -l | xargs echo "Injeção/Upload:"
        fi
        
        # Análise de tamanhos de arquivo
        echo ""
        echo "=== ANÁLISE DE TAMANHOS DE RESULTADO ==="
        grep ",SUCESSO," "$metrics_file" | cut -d',' -f9 | sort -n > /tmp/sizes_$model.txt
        if [ -s /tmp/sizes_$model.txt ]; then
            local min_size=$(head -1 /tmp/sizes_$model.txt)
            local max_size=$(tail -1 /tmp/sizes_$model.txt)
            local avg_size=$(awk '{sum+=$1} END {print sum/NR}' /tmp/sizes_$model.txt)
            
            echo "Tamanho mínimo: ${min_size} bytes"
            echo "Tamanho máximo: ${max_size} bytes"
            echo "Tamanho médio: ${avg_size} bytes"
        fi
        rm -f /tmp/sizes_$model.txt
        
    } > "$report_file"
    
    echo "✅ Relatório gerado: $report_file"
}

# Função para comparar modelos
compare_models() {
    local comparison_file="$REPORTS_DIR/models_comparison.txt"
    
    echo "🔍 Gerando comparação entre modelos..."
    
    {
        echo "=== COMPARAÇÃO ENTRE MODELOS ==="
        echo "Gerado em: $(date)"
        echo ""
        
        printf "%-15s %-10s %-10s %-10s %-10s %-12s\n" "MODELO" "SUCESSOS" "ERROS" "TIMEOUTS" "ACURÁCIA" "TEMPO_MÉDIO"
        echo "=============================================================================="
        
        for model in DistilBERT MiniLM TinyBERT; do
            local metrics_file="$RESULTS_DIR/$model/metrics_summary.txt"
            
            if [ -f "$metrics_file" ]; then
                local success_count=$(grep -c ",SUCESSO," "$metrics_file" 2>/dev/null || echo "0")
                local error_count=$(grep -c ",ERRO," "$metrics_file" 2>/dev/null || echo "0")
                local timeout_count=$(grep -c ",TIMEOUT," "$metrics_file" 2>/dev/null || echo "0")
                
                # Calcular acurácia média
                local total_correct=$(grep ",SUCESSO," "$metrics_file" | cut -d',' -f4 | awk '{sum+=$1} END {print sum+0}')
                local total_incorrect=$(grep ",SUCESSO," "$metrics_file" | cut -d',' -f5 | awk '{sum+=$1} END {print sum+0}')
                local total_predictions=$((total_correct + total_incorrect))
                local accuracy="0%"
                if [ $total_predictions -gt 0 ]; then
                    accuracy=$(echo "scale=1; $total_correct / $total_predictions * 100" | bc 2>/dev/null || echo "0")"%"
                fi
                
                # Calcular tempo médio
                local avg_time="N/A"
                grep ",SUCESSO," "$metrics_file" | cut -d',' -f3 | grep -v "N/A" | sort -n > /tmp/times_$model.txt
                if [ -s /tmp/times_$model.txt ]; then
                    avg_time=$(awk '{sum+=$1} END {printf "%.1fs", sum/NR}' /tmp/times_$model.txt)
                fi
                rm -f /tmp/times_$model.txt
                
                printf "%-15s %-10s %-10s %-10s %-10s %-12s\n" "$model" "$success_count" "$error_count" "$timeout_count" "$accuracy" "$avg_time"
            else
                printf "%-15s %-10s %-10s %-10s %-10s %-12s\n" "$model" "N/A" "N/A" "N/A" "N/A" "N/A"
            fi
        done
        
        echo ""
        echo "=== ANÁLISE CONSOLIDADA ==="
        
        # Encontrar modelo com melhor performance
        local best_accuracy=0
        local best_model=""
        
        for model in DistilBERT MiniLM TinyBERT; do
            local metrics_file="$RESULTS_DIR/$model/metrics_summary.txt"
            if [ -f "$metrics_file" ]; then
                local total_correct=$(grep ",SUCESSO," "$metrics_file" | cut -d',' -f4 | awk '{sum+=$1} END {print sum+0}')
                local total_incorrect=$(grep ",SUCESSO," "$metrics_file" | cut -d',' -f5 | awk '{sum+=$1} END {print sum+0}')
                local total_predictions=$((total_correct + total_incorrect))
                
                if [ $total_predictions -gt 0 ]; then
                    local accuracy_val=$(echo "scale=2; $total_correct / $total_predictions * 100" | bc 2>/dev/null || echo "0")
                    local accuracy_int=$(echo "$accuracy_val" | cut -d'.' -f1)
                    
                    if [ "$accuracy_int" -gt "$best_accuracy" ]; then
                        best_accuracy=$accuracy_int
                        best_model=$model
                    fi
                fi
            fi
        done
        
        if [ -n "$best_model" ]; then
            echo "🏆 Modelo com melhor acurácia: $best_model (${best_accuracy}%)"
        fi
        
        # Arquivos mais problemáticos (com falhas em todos os modelos)
        echo ""
        echo "=== ARQUIVOS MAIS PROBLEMÁTICOS ==="
        echo "Arquivos que falharam em múltiplos modelos:"
        
        # Encontrar arquivos com erro em pelo menos 2 modelos
        local all_errors=""
        for model in DistilBERT MiniLM TinyBERT; do
            local metrics_file="$RESULTS_DIR/$model/metrics_summary.txt"
            if [ -f "$metrics_file" ]; then
                all_errors="$all_errors $(grep ",ERRO," "$metrics_file" | cut -d',' -f1)"
            fi
        done
        
        echo "$all_errors" | tr ' ' '\n' | sort | uniq -c | sort -nr | head -10 | while read count file; do
            if [ "$count" -ge 2 ]; then
                echo "  $file: falhou em $count modelos"
            fi
        done
        
    } > "$comparison_file"
    
    echo "✅ Comparação gerada: $comparison_file"
}

# Função para gerar relatório de arquivos faltantes
generate_missing_report() {
    local missing_file="$REPORTS_DIR/missing_files_report.txt"
    
    echo "📋 Gerando relatório de arquivos faltantes..."
    
    {
        echo "=== RELATÓRIO DE ARQUIVOS FALTANTES ==="
        echo "Gerado em: $(date)"
        echo ""
        
        local csv_files=($(find "./data" -name "*.csv" -exec basename {} \; | sort))
        local total_csv=${#csv_files[@]}
        
        echo "Total de arquivos CSV: $total_csv"
        echo ""
        
        for model in DistilBERT MiniLM TinyBERT; do
            echo "=== $model ==="
            
            local model_dir_path=""
            case $model in
                "DistilBERT") model_dir_path="./DistilBERT" ;;
                "MiniLM") model_dir_path="./MiniLM" ;;
                "TinyBERT") model_dir_path="./TinyBERT" ;;
            esac
            
            local missing_files=()
            local processed_count=0
            
            for csv_file in "${csv_files[@]}"; do
                local expected_result="result-$(echo $model | tr '[:upper:]' '[:lower:]')-part-$csv_file.txt"
                if [ -f "$model_dir_path/$expected_result" ] && [ -s "$model_dir_path/$expected_result" ]; then
                    processed_count=$((processed_count + 1))
                else
                    missing_files+=("$csv_file")
                fi
            done
            
            local completeness=$((processed_count * 100 / total_csv))
            echo "Processados: $processed_count/$total_csv (${completeness}%)"
            
            if [ ${#missing_files[@]} -eq 0 ]; then
                echo "Status: ✅ COMPLETO"
            else
                echo "Status: ⚠️  INCOMPLETO (${#missing_files[@]} arquivos faltando)"
                echo ""
                echo "Arquivos faltantes:"
                for missing in "${missing_files[@]}"; do
                    echo "  - $missing"
                done
            fi
            echo ""
        done
        
        # Arquivos que estão faltando em TODOS os modelos
        echo "=== ARQUIVOS FALTANDO EM TODOS OS MODELOS ==="
        local missing_in_all=()
        
        for csv_file in "${csv_files[@]}"; do
            local missing_count=0
            
            for model in DistilBERT MiniLM TinyBERT; do
                local model_dir_path=""
                case $model in
                    "DistilBERT") model_dir_path="./DistilBERT" ;;
                    "MiniLM") model_dir_path="./MiniLM" ;;
                    "TinyBERT") model_dir_path="./TinyBERT" ;;
                esac
                
                local expected_result="result-$(echo $model | tr '[:upper:]' '[:lower:]')-part-$csv_file.txt"
                if [ ! -f "$model_dir_path/$expected_result" ] || [ ! -s "$model_dir_path/$expected_result" ]; then
                    missing_count=$((missing_count + 1))
                fi
            done
            
            if [ $missing_count -eq 3 ]; then
                missing_in_all+=("$csv_file")
            fi
        done
        
        if [ ${#missing_in_all[@]} -eq 0 ]; then
            echo "✅ Nenhum arquivo está faltando em todos os modelos"
        else
            echo "⚠️  Arquivos faltando em TODOS os modelos (${#missing_in_all[@]} arquivos):"
            for missing in "${missing_in_all[@]}"; do
                echo "  - $missing"
            done
        fi
        
    } > "$missing_file"
    
    echo "✅ Relatório de faltantes gerado: $missing_file"
}

# Menu principal
show_menu() {
    echo "🎯 Escolha uma opção:"
    echo "1. Analisar métricas do DistilBERT"
    echo "2. Analisar métricas do MiniLM"
    echo "3. Analisar métricas do TinyBERT"
    echo "4. Analisar TODOS os modelos"
    echo "5. Comparar modelos"
    echo "6. Relatório de arquivos faltantes"
    echo "7. Gerar TODOS os relatórios"
    echo "8. Mostrar resumo rápido"
    echo "9. Sair"
    echo ""
}

# Função para resumo rápido
show_quick_summary() {
    echo "⚡ === RESUMO RÁPIDO ==="
    echo ""
    
    local csv_total=$(find "./data" -name "*.csv" | wc -l)
    echo "📁 Total de arquivos CSV: $csv_total"
    echo ""
    
    for model in DistilBERT MiniLM TinyBERT; do
        local metrics_file="$RESULTS_DIR/$model/metrics_summary.txt"
        
        if [ -f "$metrics_file" ]; then
            local success_count=$(grep -c ",SUCESSO," "$metrics_file" 2>/dev/null || echo "0")
            local error_count=$(grep -c ",ERRO," "$metrics_file" 2>/dev/null || echo "0")
            local completeness=$((success_count * 100 / csv_total))
            
            local total_correct=$(grep ",SUCESSO," "$metrics_file" | cut -d',' -f4 | awk '{sum+=$1} END {print sum+0}')
            local total_incorrect=$(grep ",SUCESSO," "$metrics_file" | cut -d',' -f5 | awk '{sum+=$1} END {print sum+0}')
            local total_predictions=$((total_correct + total_incorrect))
            
            local accuracy="N/A"
            if [ $total_predictions -gt 0 ]; then
                accuracy=$(echo "scale=1; $total_correct / $total_predictions * 100" | bc 2>/dev/null || echo "0")"%"
            fi
            
            echo "🔸 $model: ${completeness}% completo, $success_count sucessos, $error_count erros, acurácia: $accuracy"
        else
            echo "🔸 $model: Métricas não encontradas"
        fi
    done
}

# Verificar argumentos da linha de comando
if [ $# -eq 0 ]; then
    # Modo interativo
    while true; do
        show_menu
        read -p "Digite sua escolha (1-9): " choice
        
        case $choice in
            1) analyze_model_metrics "DistilBERT" ;;
            2) analyze_model_metrics "MiniLM" ;;
            3) analyze_model_metrics "TinyBERT" ;;
            4) 
                analyze_model_metrics "DistilBERT"
                analyze_model_metrics "MiniLM"
                analyze_model_metrics "TinyBERT"
                ;;
            5) compare_models ;;
            6) generate_missing_report ;;
            7)
                echo "🔄 Gerando TODOS os relatórios..."
                analyze_model_metrics "DistilBERT"
                analyze_model_metrics "MiniLM"
                analyze_model_metrics "TinyBERT"
                compare_models
                generate_missing_report
                echo "✅ Todos os relatórios gerados em: $REPORTS_DIR"
                ;;
            8) show_quick_summary ;;
            9) echo "👋 Saindo..."; exit 0 ;;
            *) echo "❌ Opção inválida. Tente novamente." ;;
        esac
        echo ""
        read -p "Pressione Enter para continuar..."
        clear
    done
else
    # Modo com argumentos
    case $1 in
        "--all") 
            analyze_model_metrics "DistilBERT"
            analyze_model_metrics "MiniLM"
            analyze_model_metrics "TinyBERT"
            compare_models
            generate_missing_report
            ;;
        "--distilbert") analyze_model_metrics "DistilBERT" ;;
        "--minilm") analyze_model_metrics "MiniLM" ;;
        "--tinybert") analyze_model_metrics "TinyBERT" ;;
        "--compare") compare_models ;;
        "--missing") generate_missing_report ;;
        "--summary") show_quick_summary ;;
        "--help")
            echo "Uso: $0 [OPÇÃO]"
            echo ""
            echo "Opções:"
            echo "  --all         Gerar todos os relatórios"
            echo "  --distilbert  Analisar DistilBERT"
            echo "  --minilm      Analisar MiniLM"
            echo "  --tinybert    Analisar TinyBERT"
            echo "  --compare     Comparar modelos"
            echo "  --missing     Relatório de faltantes"
            echo "  --summary     Resumo rápido"
            echo "  --help        Mostrar esta ajuda"
            ;;
        *)
            echo "❌ Argumento inválido: $1"
            echo "Use --help para ver as opções disponíveis"
            exit 1
            ;;
    esac
fi

echo ""
echo "✅ Análise concluída! Relatórios disponíveis em: $REPORTS_DIR"
