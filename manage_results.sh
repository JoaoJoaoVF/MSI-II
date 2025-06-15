#!/bin/bash

# Script para gerenciar e organizar resultados dos monitores
# Compacta, limpa e organiza arquivos de resultado

echo "=== GERENCIADOR DE RESULTADOS ==="
echo "Data/Hora: $(date)"
echo ""

# Diretórios
RESULTS_DIR="./analysis_results"
ARCHIVE_DIR="./archived_results"
BACKUP_DIR="./backup_results"

# Criar diretórios se não existirem
mkdir -p "$ARCHIVE_DIR" "$BACKUP_DIR" "$RESULTS_DIR"/{DistilBERT,MiniLM,TinyBERT}

# Função para mostrar estatísticas de arquivos
show_file_stats() {
    echo "📊 === ESTATÍSTICAS DE ARQUIVOS ==="
    echo ""
    
    for model in DistilBERT MiniLM TinyBERT; do
        echo "🔸 $model:"
        if [ -d "$model" ]; then
            total_results=$(find "$model" -name "result-*.txt" | wc -l)
            empty_results=$(find "$model" -name "result-*.txt" -size 0 | wc -l)
            valid_results=$((total_results - empty_results))
            total_size=$(du -sh "$model" 2>/dev/null | cut -f1)
            
            echo "    Total de arquivos result: $total_results"
            echo "    Arquivos vazios: $empty_results"
            echo "    Arquivos válidos: $valid_results"
            echo "    Tamanho total: $total_size"
            
            if [ $total_results -gt 0 ]; then
                echo "    Taxa de sucesso: $(( valid_results * 100 / total_results ))%"
            fi
        else
            echo "    Diretório não encontrado"
        fi
        echo ""
    done
}

# Função para limpar arquivos vazios
clean_empty_files() {
    echo "🧹 === LIMPANDO ARQUIVOS VAZIOS ==="
    echo ""
    
    local removed_count=0
    
    for model in DistilBERT MiniLM TinyBERT; do
        if [ -d "$model" ]; then
            echo "Limpando $model..."
            local empty_files=$(find "$model" -name "result-*.txt" -size 0)
            local count=$(echo "$empty_files" | grep -c "result-" 2>/dev/null || echo "0")
            
            if [ $count -gt 0 ]; then
                echo "$empty_files" | xargs rm -f
                echo "  ✅ Removidos $count arquivos vazios"
                removed_count=$((removed_count + count))
            else
                echo "  ✅ Nenhum arquivo vazio encontrado"
            fi
        fi
    done
    
    echo ""
    echo "Total de arquivos vazios removidos: $removed_count"
}

# Função para compactar arquivos grandes
compress_large_files() {
    echo "📦 === COMPACTANDO ARQUIVOS GRANDES ==="
    echo ""
    
    local compressed_count=0
    
    for model in DistilBERT MiniLM TinyBERT; do
        if [ -d "$model" ]; then
            echo "Verificando $model..."
            
            # Compactar arquivos result maiores que 10MB
            local large_files=$(find "$model" -name "result-*.txt" -size +10M)
            
            if [ -n "$large_files" ]; then
                echo "$large_files" | while read -r file; do
                    if [ -f "$file" ]; then
                        echo "  Compactando: $(basename "$file")"
                        gzip "$file"
                        compressed_count=$((compressed_count + 1))
                    fi
                done
                echo "  ✅ Arquivos grandes compactados"
            else
                echo "  ✅ Nenhum arquivo grande encontrado"
            fi
            
            # Compactar logs de ataque se existirem e forem grandes
            if [ -f "$model/attack_log.json" ]; then
                local log_size=$(stat -f%z "$model/attack_log.json" 2>/dev/null || stat -c%s "$model/attack_log.json" 2>/dev/null || echo 0)
                if [ $log_size -gt 5242880 ]; then  # 5MB
                    echo "  Compactando attack_log.json..."
                    gzip "$model/attack_log.json"
                fi
            fi
        fi
    done
    
    echo ""
    echo "Compactação concluída!"
}

# Função para criar arquivo de backup
create_backup() {
    echo "💾 === CRIANDO BACKUP ==="
    echo ""
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_file="$BACKUP_DIR/results_backup_$timestamp.tar.gz"
    
    echo "Criando backup: $backup_file"
    
    # Criar backup de todos os arquivos de resultado
    tar -czf "$backup_file" \
        */result-*.txt* \
        */attack_log.json* \
        "$RESULTS_DIR"/*/processing_summary.txt \
        2>/dev/null
    
    if [ -f "$backup_file" ]; then
        local backup_size=$(du -sh "$backup_file" | cut -f1)
        echo "✅ Backup criado com sucesso: $backup_size"
        echo "📁 Localização: $backup_file"
        
        # Manter apenas os 5 backups mais recentes
        local backup_count=$(find "$BACKUP_DIR" -name "results_backup_*.tar.gz" | wc -l)
        if [ $backup_count -gt 5 ]; then
            echo "🧹 Removendo backups antigos..."
            find "$BACKUP_DIR" -name "results_backup_*.tar.gz" | sort | head -n $((backup_count - 5)) | xargs rm -f
        fi
    else
        echo "❌ Erro ao criar backup"
        return 1
    fi
}

# Função para arquivar resultados antigos
archive_old_results() {
    echo "📚 === ARQUIVANDO RESULTADOS ANTIGOS ==="
    echo ""
    
    local days_old=7  # Arquivar arquivos mais antigos que 7 dias
    local archived_count=0
    
    for model in DistilBERT MiniLM TinyBERT; do
        if [ -d "$model" ]; then
            echo "Arquivando resultados antigos de $model..."
            mkdir -p "$ARCHIVE_DIR/$model"
            
            # Encontrar arquivos antigos
            local old_files=$(find "$model" -name "result-*.txt*" -mtime +$days_old)
            
            if [ -n "$old_files" ]; then
                echo "$old_files" | while read -r file; do
                    if [ -f "$file" ]; then
                        mv "$file" "$ARCHIVE_DIR/$model/"
                        archived_count=$((archived_count + 1))
                    fi
                done
                echo "  ✅ Arquivos antigos movidos para $ARCHIVE_DIR/$model/"
            else
                echo "  ✅ Nenhum arquivo antigo encontrado"
            fi
        fi
    done
    
    echo ""
    echo "Total de arquivos arquivados: $archived_count"
}

# Função para consolidar resumos
consolidate_summaries() {
    echo "📋 === CONSOLIDANDO RESUMOS ==="
    echo ""
    
    local consolidated_file="$RESULTS_DIR/consolidated_summary.txt"
    
    {
        echo "=== RESUMO CONSOLIDADO DE TODOS OS MODELOS ==="
        echo "Gerado em: $(date)"
        echo ""
        
        for model in DistilBERT MiniLM TinyBERT; do
            echo "================== $model =================="
            
            local summary_file="$RESULTS_DIR/$model/processing_summary.txt"
            if [ -f "$summary_file" ]; then
                cat "$summary_file"
            else
                echo "Resumo não encontrado para $model"
            fi
            echo ""
        done
        
        echo "================== ESTATÍSTICAS GERAIS =================="
        show_file_stats
        
    } > "$consolidated_file"
    
    echo "✅ Resumo consolidado criado: $consolidated_file"
}

# Menu principal
show_menu() {
    echo "🎯 Escolha uma opção:"
    echo "1. Mostrar estatísticas de arquivos"
    echo "2. Limpar arquivos vazios"
    echo "3. Compactar arquivos grandes"
    echo "4. Criar backup completo"
    echo "5. Arquivar resultados antigos (>7 dias)"
    echo "6. Consolidar resumos"
    echo "7. Limpeza completa (2+3+5)"
    echo "8. Manutenção completa (1+2+3+4+5+6)"
    echo "9. Sair"
    echo ""
}

# Verificar argumentos da linha de comando
if [ $# -eq 0 ]; then
    # Modo interativo
    while true; do
        show_menu
        read -p "Digite sua escolha (1-9): " choice
        
        case $choice in
            1)
                show_file_stats
                ;;
            2)
                clean_empty_files
                ;;
            3)
                compress_large_files
                ;;
            4)
                create_backup
                ;;
            5)
                archive_old_results
                ;;
            6)
                consolidate_summaries
                ;;
            7)
                echo "🔄 Executando limpeza completa..."
                clean_empty_files
                compress_large_files
                archive_old_results
                echo "✅ Limpeza completa concluída!"
                ;;
            8)
                echo "🔄 Executando manutenção completa..."
                show_file_stats
                clean_empty_files
                compress_large_files
                create_backup
                archive_old_results
                consolidate_summaries
                echo "✅ Manutenção completa concluída!"
                ;;
            9)
                echo "👋 Saindo..."
                exit 0
                ;;
            *)
                echo "❌ Opção inválida. Tente novamente."
                ;;
        esac
        echo ""
        read -p "Pressione Enter para continuar..."
        echo ""
    done
else
    # Modo com argumentos
    case $1 in
        "--stats")
            show_file_stats
            ;;
        "--clean")
            clean_empty_files
            ;;
        "--compress")
            compress_large_files
            ;;
        "--backup")
            create_backup
            ;;
        "--archive")
            archive_old_results
            ;;
        "--consolidate")
            consolidate_summaries
            ;;
        "--full-clean")
            clean_empty_files
            compress_large_files
            archive_old_results
            ;;
        "--full-maintenance")
            show_file_stats
            clean_empty_files
            compress_large_files
            create_backup
            archive_old_results
            consolidate_summaries
            ;;
        "--help")
            echo "Uso: $0 [OPÇÃO]"
            echo ""
            echo "Opções:"
            echo "  --stats           Mostrar estatísticas"
            echo "  --clean           Limpar arquivos vazios"
            echo "  --compress        Compactar arquivos grandes"
            echo "  --backup          Criar backup"
            echo "  --archive         Arquivar resultados antigos"
            echo "  --consolidate     Consolidar resumos"
            echo "  --full-clean      Limpeza completa"
            echo "  --full-maintenance Manutenção completa"
            echo "  --help            Mostrar esta ajuda"
            ;;
        *)
            echo "❌ Argumento inválido: $1"
            echo "Use --help para ver as opções disponíveis"
            exit 1
            ;;
    esac
fi

echo ""
echo "✅ Operação concluída!"
