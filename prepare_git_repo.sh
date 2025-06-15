#!/bin/bash

# Script para preparar o repositório Git
# Remove arquivos desnecessários e mantém apenas os essenciais
# Baseado na análise dos scripts setup_environment.sh e run_all_monitors.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

log_header() {
    echo -e "${PURPLE}[HEADER]${NC} $1"
}

echo "📦 Preparação do Repositório Git - MSI-II"
echo "=========================================="
echo "Este script irá limpar o repositório mantendo apenas arquivos essenciais"
echo "Baseado na análise dos scripts setup_environment.sh e run_all_monitors.sh"
echo ""

# Verificar se estamos no diretório correto
if [ ! -f "setup_environment.sh" ] || [ ! -f "run_all_monitors.sh" ]; then
    log_error "Erro: Não foi possível encontrar os scripts principais!"
    log_error "Certifique-se de estar no diretório raiz do projeto MSI-II"
    exit 1
fi

# Função para criar backup completo antes da limpeza
create_full_backup() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_dir="backup_full_project_$timestamp"
    
    log_info "Criando backup completo em $backup_dir..."
    
    # Criar backup excluindo apenas venv e .git para economizar espaço
    rsync -av --progress --exclude='venv/' --exclude='.git/' . "$backup_dir/"
    
    if [ -d "$backup_dir" ]; then
        log_success "Backup criado: $backup_dir"
        return 0
    else
        log_error "Falha ao criar backup!"
        return 1
    fi
}

# Função para mostrar estatísticas antes da limpeza
show_before_stats() {
    log_header "=== ESTATÍSTICAS ANTES DA LIMPEZA ==="
    echo ""
    
    local total_files=$(find . -type f ! -path './venv/*' ! -path './.git/*' | wc -l)
    local total_size=$(du -sh . 2>/dev/null | cut -f1 || echo "N/A")
    
    echo "📁 Total de arquivos: $total_files"
    echo "💾 Tamanho total: $total_size"
    echo ""
    
    echo "📊 Distribuição por pasta:"
    for dir in DistilBERT MiniLM TinyBERT data config; do
        if [ -d "$dir" ]; then
            local files=$(find "$dir" -type f | wc -l)
            local size=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "N/A")
            echo "  $dir/: $files arquivos ($size)"
        fi
    done
    echo ""
}

# Função para criar .gitignore otimizado
create_gitignore() {
    log_info "Criando .gitignore otimizado..."
    
    cat > .gitignore << 'EOF'
# Ambiente Python
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
pip-log.txt
pip-delete-this-directory.txt
*.egg-info/

# Resultados e logs
logs/
analysis_results/
analysis_reports/
archived_results/
backup_*/
*.log
*.tmp
attack_log*.json

# Arquivos de resultado de execuções
result-*.txt
*.pid

# Dados grandes (manter apenas 1-2 exemplos)
data/part-*.csv
!data/part-00002-*.csv
data/extract_attack_types.py
data/link_data_files.txt

# Arquivos temporários
*.bak
*.tmp
*~
.DS_Store
Thumbs.db

# Scripts gerados automaticamente pelo setup_environment.sh
start_distilbert.sh
start_minilm.sh
start_tinybert.sh
simulate_distilbert.sh
simulate_minilm.sh
simulate_tinybert.sh
benchmark_distilbert.sh
benchmark_minilm.sh
benchmark_tinybert.sh

# Configurações locais e específicas
config_environment.sh
device_configs.sh
install_*.sh
requirements-raspberry.txt
requirements-iot.txt
requirements_iot.txt
network-detector.service

# Notebooks de desenvolvimento (manter apenas se necessário)
*_optimization.ipynb

# Subdiretórios de configuração específica nas pastas dos modelos
DistilBERT/config/
MiniLM/config/
TinyBERT/config/

# Arquivos específicos das pastas dos modelos que são duplicados ou temporários
DistilBERT/part-*.csv
DistilBERT/result-*.txt
DistilBERT/attack_log.json
DistilBERT/test_*.py
DistilBERT/run_*.sh
DistilBERT/start_detector.sh

MiniLM/part-*.csv
MiniLM/result-*.txt
MiniLM/attack_log.json
MiniLM/minilm_attack_log.json

TinyBERT/part-*.csv
TinyBERT/result-*.txt
TinyBERT/attack_log.json
TinyBERT/tinybert_attack_log.json
EOF

    log_success ".gitignore criado com regras otimizadas"
}

# Função para limpar scripts redundantes da raiz
clean_redundant_scripts() {
    log_info "Removendo scripts redundantes da raiz..."
    
    local redundant_scripts=(
        "start_distilbert.sh"
        "start_minilm.sh"
        "start_tinybert.sh"
        "simulate_distilbert.sh"
        "simulate_minilm.sh"
        "simulate_tinybert.sh"
        "benchmark_distilbert.sh"
        "benchmark_minilm.sh"
        "benchmark_tinybert.sh"
    )
    
    local removed_count=0
    
    for script in "${redundant_scripts[@]}"; do
        if [ -f "$script" ]; then
            rm -f "$script"
            log_success "  ✓ Removido: $script (será recriado pelo setup_environment.sh)"
            removed_count=$((removed_count + 1))
        fi
    done
    
    log_success "Scripts redundantes removidos: $removed_count"
}

# Função para limpar dados desnecessários
clean_data_directory() {
    log_info "Limpando diretório data/..."
    
    if [ ! -d "data" ]; then
        log_warning "Diretório data/ não encontrado"
        return 0
    fi
    
    cd data/
    
    # Contar arquivos CSV antes
    local csv_before=$(find . -name "part-*.csv" | wc -l)
    
    # Manter apenas 1 arquivo como exemplo
    local kept_file="part-00002-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv"
    
    if [ -f "$kept_file" ]; then
        # Remover todos os outros CSVs
        find . -name "part-*.csv" ! -name "$kept_file" -delete
        log_success "  ✓ Mantido apenas: $kept_file"
    else
        # Se o arquivo exemplo não existir, manter o primeiro encontrado
        local first_csv=$(find . -name "part-*.csv" | head -1)
        if [ -n "$first_csv" ]; then
            # Renomear para o nome padrão
            mv "$first_csv" "$kept_file"
            # Remover os outros
            find . -name "part-*.csv" ! -name "$kept_file" -delete
            log_success "  ✓ Renomeado e mantido: $kept_file"
        fi
    fi
    
    # Remover scripts de processamento
    rm -f extract_attack_types.py link_data_files.txt
    
    local csv_after=$(find . -name "part-*.csv" | wc -l)
    log_success "Arquivos CSV: $csv_before → $csv_after (mantido apenas exemplo)"
    
    cd ..
}

# Função para limpar pasta de um modelo específico
clean_model_directory() {
    local model_dir=$1
    
    log_info "Limpando pasta $model_dir/..."
    
    if [ ! -d "$model_dir" ]; then
        log_warning "Diretório $model_dir/ não encontrado"
        return 0
    fi
    
    cd "$model_dir/"
    
    local removed_count=0
    
    # Arquivos a remover
    local files_to_remove=(
        "part-*.csv"                    # Dados duplicados
        "result-*.txt"                  # Resultados de execuções
        "attack_log.json"               # Logs antigos
        "*_attack_log.json"             # Logs específicos
        "*_optimization.ipynb"          # Notebooks de desenvolvimento
        "test_*.py"                     # Scripts de teste
        "run_*.sh"                      # Scripts redundantes
        "start_detector.sh"             # Scripts específicos
        "config_environment.sh"         # Configurações específicas
        "device_configs.sh"             # Configurações de dispositivo
        "install_*.sh"                  # Scripts de instalação
        "network-detector.service"      # Configuração de serviço
        "requirements-*.txt"            # Requirements específicos
        "requirements_*.txt"            # Requirements específicos
    )
    
    for pattern in "${files_to_remove[@]}"; do
        for file in $pattern; do
            if [ -f "$file" ] || [ -d "$file" ]; then
                rm -rf "$file"
                removed_count=$((removed_count + 1))
            fi
        done
    done
    
    # Remover subdiretório config se existir
    if [ -d "config" ]; then
        rm -rf "config"
        log_success "  ✓ Removido subdiretório config/"
        removed_count=$((removed_count + 1))
    fi
    
    log_success "$model_dir: $removed_count itens removidos"
    
    cd ..
}

# Função para verificar arquivos essenciais
verify_essential_files() {
    log_info "Verificando arquivos essenciais..."
    
    local missing_files=()
    
    # Arquivos essenciais da raiz
    local root_essentials=(
        "setup_environment.sh"
        "run_all_monitors.sh"
        "requirements_consolidated.txt"
        "README.md"
        "start_all_models.sh"
        "stop_all_models.sh"
        "benchmark_all_models.sh"
    )
    
    for file in "${root_essentials[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    # Verificar modelos
    for model in DistilBERT MiniLM TinyBERT; do
        if [ -d "$model" ]; then
            cd "$model"
            
            local model_essentials=("realtime_network_monitor.py" "requirements.txt")
            
            # Adicionar arquivos específicos de cada modelo
            case $model in
                "DistilBERT")
                    model_essentials+=("performance_analyzer.py" "network_attack_detector_quantized.onnx" "model_metadata.pkl")
                    ;;
                "MiniLM")
                    model_essentials+=("minilm_network_monitor.py" "minilm_attack_detector_quantized.onnx" "minilm_metadata.pkl")
                    ;;
                "TinyBERT")
                    model_essentials+=("tinybert_network_monitor.py" "tinybert_attack_detector_quantized.onnx" "tinybert_metadata.pkl")
                    ;;
            esac
            
            for file in "${model_essentials[@]}"; do
                if [ ! -f "$file" ]; then
                    missing_files+=("$model/$file")
                fi
            done
            
            cd ..
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        log_success "✅ Todos os arquivos essenciais estão presentes!"
        return 0
    else
        log_error "❌ Arquivos essenciais faltando:"
        for file in "${missing_files[@]}"; do
            echo "    - $file"
        done
        return 1
    fi
}

# Função para limpar diretórios temporários
clean_temp_directories() {
    log_info "Removendo diretórios temporários..."
    
    local temp_dirs=(
        "logs"
        "analysis_results"
        "analysis_reports"
        "archived_results"
        "backup_*"
        "__pycache__"
    )
    
    local removed_count=0
    
    for dir_pattern in "${temp_dirs[@]}"; do
        for dir in $dir_pattern; do
            if [ -d "$dir" ]; then
                rm -rf "$dir"
                log_success "  ✓ Removido: $dir/"
                removed_count=$((removed_count + 1))
            fi
        done
    done
    
    log_success "Diretórios temporários removidos: $removed_count"
}

# Função para mostrar estatísticas finais
show_after_stats() {
    log_header "=== ESTATÍSTICAS APÓS LIMPEZA ==="
    echo ""
    
    local total_files=$(find . -type f ! -path './venv/*' ! -path './.git/*' | wc -l)
    local total_size=$(du -sh . 2>/dev/null | cut -f1 || echo "N/A")
    
    echo "📁 Total de arquivos: $total_files"
    echo "💾 Tamanho total: $total_size"
    echo ""
    
    echo "📊 Arquivos essenciais por pasta:"
    for dir in DistilBERT MiniLM TinyBERT data config; do
        if [ -d "$dir" ]; then
            local files=$(find "$dir" -type f | wc -l)
            local size=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "N/A")
            echo "  $dir/: $files arquivos ($size)"
        fi
    done
    echo ""
}

# Função para gerar relatório final
generate_final_report() {
    local report_file="GIT_CLEANUP_REPORT.txt"
    
    log_info "Gerando relatório final..."
    
    {
        echo "=== RELATÓRIO DE LIMPEZA DO REPOSITÓRIO GIT ==="
        echo "Data/Hora: $(date)"
        echo ""
        echo "=== AÇÕES REALIZADAS ==="
        echo "✅ Backup completo criado"
        echo "✅ Scripts redundantes removidos"
        echo "✅ Dados desnecessários limpos"
        echo "✅ Pastas dos modelos limpas"
        echo "✅ Diretórios temporários removidos"
        echo "✅ .gitignore otimizado criado"
        echo ""
        echo "=== ARQUIVOS MANTIDOS (ESSENCIAIS) ==="
        echo ""
        echo "📁 Raiz do projeto:"
        find . -maxdepth 1 -type f ! -name ".*" | sort
        echo ""
        echo "📁 DistilBERT/:"
        if [ -d "DistilBERT" ]; then
            find DistilBERT/ -type f | sort
        fi
        echo ""
        echo "📁 MiniLM/:"
        if [ -d "MiniLM" ]; then
            find MiniLM/ -type f | sort
        fi
        echo ""
        echo "📁 TinyBERT/:"
        if [ -d "TinyBERT" ]; then
            find TinyBERT/ -type f | sort
        fi
        echo ""
        echo "📁 config/:"
        if [ -d "config" ]; then
            find config/ -type f | sort
        fi
        echo ""
        echo "📁 data/:"
        if [ -d "data" ]; then
            find data/ -type f | sort
        fi
        echo ""
        echo "=== PRÓXIMOS PASSOS ==="
        echo "1. Revisar as mudanças: git status"
        echo "2. Adicionar arquivos: git add ."
        echo "3. Remover arquivos trackeados desnecessários: git rm --cached [arquivo]"
        echo "4. Commit: git commit -m '🧹 Limpeza do repositório: mantidos apenas arquivos essenciais'"
        echo "5. Push: git push"
        echo ""
        echo "=== INFORMAÇÕES IMPORTANTES ==="
        echo "• Scripts auxiliares serão recriados pelo setup_environment.sh"
        echo "• Ambiente virtual será criado na primeira execução"
        echo "• Logs e resultados são gerados durante a execução"
        echo "• Adicione seus próprios dados na pasta data/"
        echo ""
    } > "$report_file"
    
    log_success "Relatório salvo em: $report_file"
}

# Menu principal
show_menu() {
    echo ""
    echo "🎯 Escolha uma opção:"
    echo "1. Análise prévia (mostrar o que seria removido)"
    echo "2. ⭐ Limpeza completa automática (recomendado)"
    echo "3. Limpeza passo a passo"
    echo "4. Apenas criar .gitignore"
    echo "5. Apenas verificar arquivos essenciais"
    echo "6. Mostrar estatísticas atuais"
    echo "7. Sair"
    echo ""
}

# Função para análise prévia
preview_cleanup() {
    log_header "=== ANÁLISE PRÉVIA (sem remover nada) ==="
    echo ""
    
    log_info "Scripts redundantes que serão removidos:"
    for script in start_distilbert.sh start_minilm.sh start_tinybert.sh simulate_*.sh benchmark_distilbert.sh benchmark_minilm.sh benchmark_tinybert.sh; do
        if [ -f "$script" ]; then
            echo "  🗑️  $script"
        fi
    done
    
    echo ""
    log_info "Dados que serão limpos:"
    if [ -d "data" ]; then
        local csv_count=$(find data/ -name "part-*.csv" | wc -l)
        echo "  📊 Arquivos CSV: $csv_count (manter apenas 1 exemplo)"
        if [ -f "data/extract_attack_types.py" ]; then
            echo "  🗑️  data/extract_attack_types.py"
        fi
        if [ -f "data/link_data_files.txt" ]; then
            echo "  🗑️  data/link_data_files.txt"
        fi
    fi
    
    echo ""
    log_info "Arquivos por pasta dos modelos que serão removidos:"
    for model in DistilBERT MiniLM TinyBERT; do
        if [ -d "$model" ]; then
            echo "  📁 $model/:"
            cd "$model"
            for pattern in "part-*.csv" "result-*.txt" "*attack_log.json" "*_optimization.ipynb" "test_*.py" "run_*.sh" "config_environment.sh" "device_configs.sh" "install_*.sh" "requirements-*.txt" "requirements_*.txt"; do
                for file in $pattern; do
                    if [ -f "$file" ]; then
                        echo "    🗑️  $file"
                    fi
                done
            done
            if [ -d "config" ]; then
                echo "    🗑️  config/ (diretório)"
            fi
            cd ..
        fi
    done
    
    echo ""
    log_info "Diretórios temporários que serão removidos:"
    for dir in logs analysis_results analysis_reports archived_results backup_* __pycache__; do
        if [ -d "$dir" ]; then
            echo "  🗑️  $dir/"
        fi
    done
}

# Função para limpeza completa automática
full_cleanup() {
    log_header "=== LIMPEZA COMPLETA AUTOMÁTICA ==="
    echo ""
    
    log_warning "ATENÇÃO: Esta operação irá:"
    echo "  • Criar backup completo do projeto"
    echo "  • Remover scripts redundantes"
    echo "  • Limpar dados desnecessários"
    echo "  • Limpar pastas dos modelos"
    echo "  • Remover diretórios temporários"
    echo "  • Criar .gitignore otimizado"
    echo ""
    
    read -p "Deseja continuar? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Operação cancelada."
        return 0
    fi
    
    show_before_stats
    
    # Executar limpeza passo a passo
    create_full_backup || {
        log_error "Falha ao criar backup. Abortando limpeza."
        return 1
    }
    
    create_gitignore
    clean_redundant_scripts
    clean_data_directory
    
    for model in DistilBERT MiniLM TinyBERT; do
        clean_model_directory "$model"
    done
    
    clean_temp_directories
    
    # Verificar se tudo está correto
    if verify_essential_files; then
        show_after_stats
        generate_final_report
        
        log_success "🎉 Limpeza completa realizada com sucesso!"
        echo ""
        log_info "📋 Próximos passos:"
        echo "  1. Revisar o relatório: cat GIT_CLEANUP_REPORT.txt"
        echo "  2. Verificar mudanças: git status"
        echo "  3. Adicionar arquivos: git add ."
        echo "  4. Commit: git commit -m '🧹 Limpeza do repositório'"
        echo "  5. Push: git push"
    else
        log_error "❌ Verificação falhou! Alguns arquivos essenciais estão faltando."
        log_warning "Recomendação: Restaurar do backup e verificar o projeto."
    fi
}

# Função principal
main() {
    echo "📁 Diretório atual: $(pwd)"
    echo "📊 Total de arquivos: $(find . -type f ! -path './venv/*' ! -path './.git/*' | wc -l)"
    echo ""
    
    # Verificar argumentos da linha de comando
    if [ $# -eq 0 ]; then
        # Modo interativo
        while true; do
            show_menu
            read -p "Digite sua escolha (1-7): " choice
            
            case $choice in
                1)
                    preview_cleanup
                    ;;
                2)
                    full_cleanup
                    ;;
                3)
                    log_info "Modo passo a passo não implementado ainda"
                    log_info "Use a opção 2 para limpeza completa automática"
                    ;;
                4)
                    create_gitignore
                    ;;
                5)
                    verify_essential_files
                    ;;
                6)
                    show_before_stats
                    ;;
                7)
                    log_info "Saindo..."
                    exit 0
                    ;;
                *)
                    log_error "Opção inválida. Digite um número de 1 a 7."
                    ;;
            esac
        done
    else
        # Modo com argumentos
        case $1 in
            "--preview"|"-p")
                preview_cleanup
                ;;
            "--full"|"-f")
                full_cleanup
                ;;
            "--gitignore"|"-g")
                create_gitignore
                ;;
            "--verify"|"-v")
                verify_essential_files
                ;;
            "--stats"|"-s")
                show_before_stats
                ;;
            "--help"|"-h")
                echo "Uso: $0 [OPÇÃO]"
                echo ""
                echo "Opções:"
                echo "  --preview     Mostrar o que seria removido"
                echo "  --full        Limpeza completa automática"
                echo "  --gitignore   Criar apenas .gitignore"
                echo "  --verify      Verificar arquivos essenciais"
                echo "  --stats       Mostrar estatísticas atuais"
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
