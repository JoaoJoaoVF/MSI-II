#!/bin/bash

# Script para limpar arquivos desnecessários das pastas dos modelos
# Mantém apenas os arquivos essenciais identificados pelos scripts setup_environment.sh e run_all_monitors.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "🧹 Script de Limpeza dos Modelos"
echo "================================"
echo "Este script irá remover arquivos desnecessários das pastas dos modelos"
echo "Baseado na análise dos scripts setup_environment.sh e run_all_monitors.sh"
echo ""

# Função para criar backup antes da limpeza
create_backup() {
    local model_dir=$1
    local backup_dir="backup_$(basename $model_dir)_$(date +%Y%m%d_%H%M%S)"
    
    log_info "Criando backup de $model_dir em $backup_dir..."
    cp -r "$model_dir" "$backup_dir"
    log_success "Backup criado: $backup_dir"
}

# Função para limpar um modelo específico
cleanup_model() {
    local model_name=$1
    local model_dir=$2
    
    log_info "Limpando pasta do modelo: $model_name"
    
    if [ ! -d "$model_dir" ]; then
        log_error "Diretório $model_dir não encontrado!"
        return 1
    fi
    
    cd "$model_dir"
    
    # Arquivos que podem ser removidos (comuns a todos os modelos)
    files_to_remove=(
        "part-*.csv"                    # Dados duplicados da pasta data/
        "result-*.txt"                  # Resultados de execuções anteriores
        "attack_log.json"               # Logs antigos
        "*_attack_log.json"             # Logs específicos antigos
        "*_optimization.ipynb"          # Notebooks de otimização
        "test_*.py"                     # Scripts de teste
        "run_*.sh"                      # Scripts redundantes
        "start_detector.sh"             # Scripts específicos redundantes
        "network-detector.service"      # Configurações de serviço
        "config_environment.sh"         # Configurações específicas
        "requirements-raspberry.txt"    # Requirements específicos
        "requirements_iot.txt"          # Requirements IoT
        "device_configs.sh"             # Configurações de dispositivo
        "install_*.sh"                  # Scripts de instalação específicos
        "*.log"                         # Arquivos de log antigos
        "*.tmp"                         # Arquivos temporários
        "*.bak"                         # Arquivos de backup
    )
    
    local removed_count=0
    local total_size_before=$(du -s . 2>/dev/null | cut -f1 || echo "0")
    
    log_info "Removendo arquivos desnecessários de $model_name..."
    
    for pattern in "${files_to_remove[@]}"; do
        for file in $pattern; do
            if [ -f "$file" ] || [ -d "$file" ]; then
                local file_size=$(du -s "$file" 2>/dev/null | cut -f1 || echo "0")
                rm -rf "$file"
                log_success "  ✓ Removido: $file (${file_size}KB)"
                removed_count=$((removed_count + 1))
            fi
        done
    done
    
    # Remover diretórios vazios
    find . -type d -empty -delete 2>/dev/null || true
    
    local total_size_after=$(du -s . 2>/dev/null | cut -f1 || echo "0")
    local size_saved=$((total_size_before - total_size_after))
    
    log_success "$model_name: $removed_count arquivos removidos, ${size_saved}KB economizados"
    
    cd ..
}

# Função para verificar arquivos essenciais
verify_essential_files() {
    local model_name=$1
    local model_dir=$2
    
    log_info "Verificando arquivos essenciais em $model_name..."
    
    cd "$model_dir"
    
    # Definir arquivos essenciais baseado no modelo
    local essential_files=()
    
    case $model_name in
        "DistilBERT")
            essential_files=(
                "realtime_network_monitor.py"
                "performance_analyzer.py"
                "network_attack_detector_quantized.onnx"
                "model_metadata.pkl"
                "requirements.txt"
            )
            ;;
        "MiniLM")
            essential_files=(
                "realtime_network_monitor.py"
                "minilm_network_monitor.py"
                "minilm_attack_detector_quantized.onnx"
                "minilm_metadata.pkl"
                "requirements.txt"
            )
            ;;
        "TinyBERT")
            essential_files=(
                "realtime_network_monitor.py"
                "tinybert_network_monitor.py"
                "tinybert_attack_detector_quantized.onnx"
                "tinybert_metadata.pkl"
                "requirements.txt"
            )
            ;;
    esac
    
    local missing_files=()
    local present_files=()
    
    for file in "${essential_files[@]}"; do
        if [ -f "$file" ]; then
            present_files+=("$file")
            log_success "  ✓ $file"
        else
            missing_files+=("$file")
            log_error "  ✗ $file (FALTANDO!)"
        fi
    done
    
    echo ""
    log_info "Resumo $model_name:"
    echo "  ✅ Arquivos presentes: ${#present_files[@]}/${#essential_files[@]}"
    echo "  ❌ Arquivos faltando: ${#missing_files[@]}"
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        log_warning "ATENÇÃO: Arquivos essenciais faltando em $model_name!"
        for missing in "${missing_files[@]}"; do
            echo "    - $missing"
        done
        return 1
    fi
    
    cd ..
    return 0
}

# Função para mostrar estatísticas antes e depois
show_disk_usage() {
    echo ""
    log_info "Uso de disco por modelo:"
    
    for model in DistilBERT MiniLM TinyBERT; do
        if [ -d "$model" ]; then
            local size=$(du -sh "$model" 2>/dev/null | cut -f1 || echo "N/A")
            local files=$(find "$model" -type f | wc -l)
            echo "  📁 $model: $size ($files arquivos)"
        fi
    done
    echo ""
}

# Menu de opções
show_menu() {
    echo ""
    echo "🎯 Escolha uma opção:"
    echo "1. Analisar arquivos (apenas mostrar o que seria removido)"
    echo "2. Limpar DistilBERT apenas"
    echo "3. Limpar MiniLM apenas"
    echo "4. Limpar TinyBERT apenas"
    echo "5. ⭐ Limpar todos os modelos (recomendado)"
    echo "6. Verificar arquivos essenciais"
    echo "7. Mostrar uso de disco"
    echo "8. Criar backup de todos os modelos"
    echo "9. Sair"
    echo ""
}

# Função para análise sem remoção
analyze_only() {
    log_info "=== ANÁLISE DOS ARQUIVOS (sem remoção) ==="
    echo ""
    
    for model in DistilBERT MiniLM TinyBERT; do
        if [ -d "$model" ]; then
            echo "🔍 Analisando $model:"
            cd "$model"
            
            # Contar arquivos que seriam removidos
            local removable_count=0
            local removable_size=0
            
            patterns=("part-*.csv" "result-*.txt" "attack_log.json" "*_attack_log.json" 
                     "*_optimization.ipynb" "test_*.py" "run_*.sh" "start_detector.sh"
                     "network-detector.service" "config_environment.sh" "requirements-raspberry.txt"
                     "requirements_iot.txt" "device_configs.sh" "install_*.sh")
            
            for pattern in "${patterns[@]}"; do
                for file in $pattern; do
                    if [ -f "$file" ] || [ -d "$file" ]; then
                        local file_size=$(du -s "$file" 2>/dev/null | cut -f1 || echo "0")
                        echo "  🗑️  $file (${file_size}KB)"
                        removable_count=$((removable_count + 1))
                        removable_size=$((removable_size + file_size))
                    fi
                done
            done
            
            echo "  📊 Total removível: $removable_count arquivos, ${removable_size}KB"
            cd ..
            echo ""
        fi
    done
}

# Função principal
main() {
    echo "📁 Diretório atual: $(pwd)"
    echo ""
    
    # Verificar se estamos no diretório correto
    if [ ! -d "DistilBERT" ] || [ ! -d "MiniLM" ] || [ ! -d "TinyBERT" ]; then
        log_error "Erro: Não foi possível encontrar as pastas dos modelos!"
        log_error "Certifique-se de estar no diretório raiz do projeto MSI-II"
        exit 1
    fi
    
    show_disk_usage
    
    # Verificar argumentos da linha de comando
    if [ $# -eq 0 ]; then
        # Modo interativo
        while true; do
            show_menu
            read -p "Digite sua escolha (1-9): " choice
            
            case $choice in
                1)
                    analyze_only
                    ;;
                2)
                    read -p "Criar backup antes da limpeza? (y/N): " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        create_backup "DistilBERT"
                    fi
                    cleanup_model "DistilBERT" "DistilBERT"
                    verify_essential_files "DistilBERT" "DistilBERT"
                    ;;
                3)
                    read -p "Criar backup antes da limpeza? (y/N): " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        create_backup "MiniLM"
                    fi
                    cleanup_model "MiniLM" "MiniLM"
                    verify_essential_files "MiniLM" "MiniLM"
                    ;;
                4)
                    read -p "Criar backup antes da limpeza? (y/N): " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        create_backup "TinyBERT"
                    fi
                    cleanup_model "TinyBERT" "TinyBERT"
                    verify_essential_files "TinyBERT" "TinyBERT"
                    ;;
                5)
                    echo ""
                    log_warning "ATENÇÃO: Isso irá limpar TODOS os modelos!"
                    read -p "Tem certeza? (y/N): " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        read -p "Criar backup antes da limpeza? (y/N): " -n 1 -r
                        echo
                        if [[ $REPLY =~ ^[Yy]$ ]]; then
                            for model in DistilBERT MiniLM TinyBERT; do
                                create_backup "$model"
                            done
                        fi
                        
                        cleanup_model "DistilBERT" "DistilBERT"
                        cleanup_model "MiniLM" "MiniLM"
                        cleanup_model "TinyBERT" "TinyBERT"
                        
                        echo ""
                        log_info "Verificando arquivos essenciais após limpeza..."
                        verify_essential_files "DistilBERT" "DistilBERT"
                        verify_essential_files "MiniLM" "MiniLM"
                        verify_essential_files "TinyBERT" "TinyBERT"
                        
                        show_disk_usage
                        log_success "🎉 Limpeza de todos os modelos concluída!"
                    fi
                    ;;
                6)
                    verify_essential_files "DistilBERT" "DistilBERT"
                    verify_essential_files "MiniLM" "MiniLM"
                    verify_essential_files "TinyBERT" "TinyBERT"
                    ;;
                7)
                    show_disk_usage
                    ;;
                8)
                    for model in DistilBERT MiniLM TinyBERT; do
                        if [ -d "$model" ]; then
                            create_backup "$model"
                        fi
                    done
                    ;;
                9)
                    log_info "Saindo..."
                    exit 0
                    ;;
                *)
                    log_error "Opção inválida. Digite um número de 1 a 9."
                    ;;
            esac
        done
    else
        # Modo com argumentos
        case $1 in
            "--analyze"|"-a")
                analyze_only
                ;;
            "--clean-all"|"-ca")
                for model in DistilBERT MiniLM TinyBERT; do
                    cleanup_model "$model" "$model"
                done
                ;;
            "--verify"|"-v")
                for model in DistilBERT MiniLM TinyBERT; do
                    verify_essential_files "$model" "$model"
                done
                ;;
            "--backup"|"-b")
                for model in DistilBERT MiniLM TinyBERT; do
                    create_backup "$model"
                done
                ;;
            "--help"|"-h")
                echo "Uso: $0 [OPÇÃO]"
                echo ""
                echo "Opções:"
                echo "  --analyze     Analisar arquivos sem remover"
                echo "  --clean-all   Limpar todos os modelos"
                echo "  --verify      Verificar arquivos essenciais"
                echo "  --backup      Criar backup de todos os modelos"
                echo "  --help        Mostrar esta ajuda"
                echo ""
                echo "Sem argumentos: modo interativo"
                ;;
            *)
                log_error "Argumento inválido: $1"
                echo "Use --help para ver as opções disponíveis"
                exit 1
                ;;
        esac
    fi
}

main "$@"
