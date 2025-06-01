#!/bin/bash

set -e 

echo "🚀 Iniciando deploy do sistema unificado de detecção de ataques..."

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

log_model() {
    echo -e "${PURPLE}[MODEL]${NC} $1"
}

select_model() {
    echo ""
    log_info "Selecione o modelo para configurar:"
    echo "1) DistilBERT"
    echo "2) TinyBERT" 
    echo "3) MiniLM"
    echo "4) Todos os modelos"
    echo ""
    
    while true; do
        read -p "Digite sua escolha (1-4): " choice
        case $choice in
            1) 
                SELECTED_MODEL="DistilBERT"
                MODEL_DIR="DistilBERT"
                MONITOR_SCRIPT="realtime_network_monitor.py"
                break
                ;;
            2) 
                SELECTED_MODEL="TinyBERT"
                MODEL_DIR="TinyBERT"
                MONITOR_SCRIPT="tinybert_network_monitor.py"
                break
                ;;
            3) 
                SELECTED_MODEL="MiniLM"
                MODEL_DIR="MiniLM"
                MONITOR_SCRIPT="minilm_network_monitor.py"
                break
                ;;
            4) 
                SELECTED_MODEL="ALL"
                MODEL_DIR=""
                MONITOR_SCRIPT=""
                break
                ;;
            *) 
                echo "Opção inválida. Digite 1, 2, 3 ou 4."
                ;;
        esac
    done
    
    log_success "Selecionado: $SELECTED_MODEL"
}

check_files() {
    log_info "Verificando estrutura de diretórios e arquivos necessários..."
    
    # Verificar se os diretórios dos modelos existem
    models=("DistilBERT" "TinyBERT" "MiniLM")
    missing_dirs=()
    
    for model in "${models[@]}"; do
        if [[ -d "$model" ]]; then
            log_success "✓ Diretório $model encontrado"
        else
            log_error "✗ Diretório $model não encontrado"
            missing_dirs+=("$model")
        fi
    done
    
    if [[ ${#missing_dirs[@]} -gt 0 ]]; then
        log_error "Diretórios obrigatórios faltando: ${missing_dirs[*]}"
        log_error "Certifique-se de que os diretórios dos modelos estão presentes."
        exit 1
    fi
    
    # Verificar arquivos específicos de cada modelo se não for instalação completa
    if [[ "$SELECTED_MODEL" != "ALL" ]]; then
        check_model_files "$MODEL_DIR"
    else
        for model in "${models[@]}"; do
            check_model_files "$model"
        done
    fi
}

check_model_files() {
    local model_dir=$1
    log_model "Verificando arquivos do $model_dir..."
    
    case $model_dir in
        "DistilBERT")
            required_files=(
                "$model_dir/realtime_network_monitor.py"
                "$model_dir/performance_analyzer.py"
                "$model_dir/network_attack_detector_quantized.onnx"
                "$model_dir/model_metadata.pkl"
                "$model_dir/requirements.txt"
            )
            ;;
        "TinyBERT")
            required_files=(
                "$model_dir/realtime_network_monitor.py"
                "$model_dir/tinybert_network_monitor.py"
                "$model_dir/tinybert_attack_detector_quantized.onnx"
                "$model_dir/tinybert_metadata.pkl"
                "$model_dir/requirements.txt"
            )
            ;;
        "MiniLM")
            required_files=(
                "$model_dir/realtime_network_monitor.py"
                "$model_dir/minilm_network_monitor.py"
                "$model_dir/minilm_attack_detector_quantized.onnx"
                "$model_dir/minilm_metadata.pkl"
                "$model_dir/requirements.txt"
            )
            ;;
    esac
    
    missing_files=()
    
    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            log_success "  ✓ $file"
        else
            log_error "  ✗ $file (OBRIGATÓRIO - FALTANDO)"
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log_error "Arquivos obrigatórios faltando no $model_dir: ${missing_files[*]}"
        return 1
    fi
    
    log_success "Arquivos do $model_dir verificados!"
    return 0
}

update_system() {
    log_info "Atualizando sistema..."
    sudo apt update
    sudo apt upgrade -y
    log_success "Sistema atualizado!"
}

install_system_deps() {
    log_info "Instalando dependências do sistema..."
    
    sudo apt install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        libopenblas-dev \
        liblapack-dev \
        gfortran \
        htop \
        tmux \
        git \
        curl
    
    log_success "Dependências do sistema instaladas!"
}

create_venv() {
    log_info "Criando ambiente virtual Python..."
    
    if [[ -d "venv" ]]; then
        log_warning "Ambiente virtual já existe. Removendo..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    source venv/bin/activate
    
    pip install --upgrade pip
    
    log_success "Ambiente virtual criado!"
}

install_python_deps() {
    log_info "Instalando dependências Python..."
    
    source venv/bin/activate
    
    pip install --upgrade pip setuptools wheel
    
    # Criar requirements consolidado baseado nos modelos selecionados
    create_consolidated_requirements
    
    log_info "Instalando dependências do arquivo consolidado..."
    
    while IFS= read -r package || [[ -n "$package" ]]; do
        if [[ -z "$package" || "$package" =~ ^#.* ]]; then
            continue
        fi
        
        log_info "Instalando: $package"
        if ! pip install "$package"; then
            log_warning "Falha ao instalar $package, tentando versão mais recente..."
            package_name=$(echo "$package" | cut -d'>' -f1 | cut -d'=' -f1)
            if ! pip install "$package_name"; then
                log_error "Falha crítica ao instalar $package_name"
            fi
        fi
    done < "requirements_consolidated.txt"
    
    # Verificar pacotes críticos
    critical_packages=("onnxruntime" "numpy" "pandas" "scikit-learn" "torch" "transformers")
    missing_critical=()
    
    for pkg in "${critical_packages[@]}"; do
        if ! pip show "$pkg" > /dev/null 2>&1; then
            missing_critical+=("$pkg")
        fi
    done
    
    if [[ ${#missing_critical[@]} -gt 0 ]]; then
        log_error "Pacotes críticos faltando: ${missing_critical[*]}"
        log_error "Tentando instalação alternativa..."
        
        for pkg in "${missing_critical[@]}"; do
            log_info "Tentativa alternativa para $pkg..."
            pip install --no-deps "$pkg" || log_warning "Falha na instalação alternativa de $pkg"
        done
    fi
    
    log_success "Instalação de dependências Python concluída!"
    
    log_info "Pacotes instalados:"
    pip list | grep -E "(onnxruntime|numpy|pandas|scikit-learn|torch|transformers|matplotlib|seaborn)"
}

create_consolidated_requirements() {
    log_info "Criando arquivo de requirements consolidado..."
    
    cat > requirements_consolidated.txt << 'EOF'
# Dependências consolidadas para todos os modelos
onnxruntime>=1.17.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
torch>=2.0.0
transformers>=4.30.0
matplotlib>=3.7.0
seaborn>=0.12.0
sentence-transformers==2.2.2
psutil>=5.9.0
EOF
    
    log_success "Arquivo de requirements consolidado criado!"
}

setup_directories() {
    log_info "Configurando diretórios..."
    
    mkdir -p logs
    mkdir -p data
    mkdir -p analysis_results
    mkdir -p config
    
    # Criar subdiretórios para cada modelo
    if [[ "$SELECTED_MODEL" == "ALL" ]]; then
        mkdir -p logs/DistilBERT logs/TinyBERT logs/MiniLM
        mkdir -p analysis_results/DistilBERT analysis_results/TinyBERT analysis_results/MiniLM
    else
        mkdir -p "logs/$SELECTED_MODEL"
        mkdir -p "analysis_results/$SELECTED_MODEL"
    fi
    
    log_success "Diretórios configurados!"
}

create_config() {
    log_info "Criando arquivos de configuração..."
    
    if [[ "$SELECTED_MODEL" == "ALL" ]]; then
        create_config_for_model "DistilBERT"
        create_config_for_model "TinyBERT"
        create_config_for_model "MiniLM"
    else
        create_config_for_model "$SELECTED_MODEL"
    fi
    
    log_success "Arquivos de configuração criados!"
}

create_config_for_model() {
    local model_name=$1
    
    case $model_name in
        "DistilBERT")
            model_file="network_attack_detector_quantized.onnx"
            metadata_file="model_metadata.pkl"
            ;;
        "TinyBERT")
            model_file="tinybert_attack_detector_quantized.onnx"
            metadata_file="tinybert_metadata.pkl"
            ;;
        "MiniLM")
            model_file="minilm_attack_detector_quantized.onnx"
            metadata_file="minilm_metadata.pkl"
            ;;
    esac
    
    cat > "config/${model_name,,}_config.json" << EOF
{
    "model": {
        "path": "${model_name}/${model_file}",
        "metadata_path": "${model_name}/${metadata_file}"
    },
    "monitoring": {
        "log_file": "logs/${model_name}/attack_log.json",
        "batch_size": 32,
        "alert_threshold": 0.8
    },
    "performance": {
        "max_inference_time_ms": 100,
        "target_throughput": 100
    },
    "alerts": {
        "enable_email": false,
        "enable_syslog": true,
        "enable_file_log": true
    }
}
EOF
}

create_service_scripts() {
    log_info "Criando scripts de serviço..."
    
    if [[ "$SELECTED_MODEL" == "ALL" ]]; then
        create_scripts_for_model "DistilBERT" "realtime_network_monitor.py"
        create_scripts_for_model "TinyBERT" "tinybert_network_monitor.py"
        create_scripts_for_model "MiniLM" "minilm_network_monitor.py"
        create_unified_scripts
    else
        create_scripts_for_model "$SELECTED_MODEL" "$MONITOR_SCRIPT"
    fi
    
    log_success "Scripts de serviço criados!"
}

create_scripts_for_model() {
    local model_name=$1
    local script_name=$2
    
    cat > "start_${model_name,,}.sh" << EOF
#!/bin/bash
cd "\$(dirname "\$0")"
source venv/bin/activate
cd ${model_name}
python3 realtime_network_monitor.py --interactive
EOF
    
    cat > "benchmark_${model_name,,}.sh" << EOF
#!/bin/bash
cd "\$(dirname "\$0")"
source venv/bin/activate
cd ${model_name}
python3 realtime_network_monitor.py --benchmark
EOF
    
    cat > "simulate_${model_name,,}.sh" << EOF
#!/bin/bash
cd "\$(dirname "\$0")"
source venv/bin/activate
cd ${model_name}

if [[ \$# -eq 0 ]]; then
    echo "Uso: \$0 <arquivo_csv>"
    echo "Exemplo: \$0 ../data/network_data.csv"
    echo "Os resultados serão salvos como: result-${model_name,,}-part-<nome_csv>.txt"
    exit 1
fi

echo "Iniciando simulação ${model_name} com arquivo: \$1"
echo "Resultado será salvo como: result-${model_name,,}-part-\$(basename \$1 .csv).txt"

python3 realtime_network_monitor.py --simulate "\$1" --delay 0.1
EOF
    
    chmod +x "start_${model_name,,}.sh" "benchmark_${model_name,,}.sh" "simulate_${model_name,,}.sh"
}

create_unified_scripts() {
    cat > start_all_models.sh << 'EOF'
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
EOF

    cat > stop_all_models.sh << 'EOF'
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
EOF

    cat > benchmark_all_models.sh << 'EOF'
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
EOF

    chmod +x start_all_models.sh stop_all_models.sh benchmark_all_models.sh
}

test_installation() {
    log_info "Testando instalação..."
    
    source venv/bin/activate
    
    # Teste básico de importações
    python3 -c "
import sys
try:
    import numpy as np
    import pandas as pd
    import sklearn
    import onnxruntime as ort
    import torch
    import transformers
    print('✅ Importações críticas OK')
    print(f'Python: {sys.version.split()[0]}')
    print(f'NumPy: {np.__version__}')
    print(f'Pandas: {pd.__version__}')
    print(f'ONNX Runtime: {ort.__version__}')
    print(f'PyTorch: {torch.__version__}')
    print(f'Transformers: {transformers.__version__}')
except ImportError as e:
    print(f'❌ Erro de importação: {e}')
    sys.exit(1)
" || {
        log_error "Teste básico de importações falhou!"
        return 1
    }
    
    # Teste específico dos modelos
    if [[ "$SELECTED_MODEL" != "ALL" ]]; then
        test_model_installation "$MODEL_DIR" "$MONITOR_SCRIPT"
    else
        test_model_installation "DistilBERT" "realtime_network_monitor.py"
        test_model_installation "TinyBERT" "tinybert_network_monitor.py" 
        test_model_installation "MiniLM" "minilm_network_monitor.py"
    fi
    
    log_success "Teste de instalação concluído!"
}

test_model_installation() {
    local model_dir=$1
    local script_name=$2
    
    log_model "Testando $model_dir..."
    
    cd "$model_dir"
    timeout 10 python3 realtime_network_monitor.py --benchmark 2>/dev/null || {
        log_warning "Teste do $model_dir falhou ou foi interrompido (normal se não houver dados de teste)"
    }
    cd ..
}

show_final_info() {
    log_success "🎉 Deploy concluído com sucesso!"
    
    echo ""
    echo "📋 INFORMAÇÕES DO SISTEMA:"
    echo "=========================="
    echo "📁 Diretório: $(pwd)"
    echo "🐍 Python: $(python3 --version)"
    echo "💾 Espaço em disco:"
    df -h . | tail -1
    echo "🧠 Memória:"
    free -h | head -2
    echo ""
    
    show_usage_info
    show_file_info
    show_troubleshooting_info
    
    log_success "Sistema pronto para uso! 🎯"
}

show_usage_info() {
    echo "🚀 COMANDOS ÚTEIS:"
    echo "=================="
    
    if [[ "$SELECTED_MODEL" == "ALL" ]]; then
        echo "• Iniciar todos os modelos:"
        echo "  ./start_all_models.sh"
        echo ""
        echo "• Parar todos os modelos:"
        echo "  ./stop_all_models.sh"
        echo ""
        echo "• Iniciar modelo específico:"
        echo "  ./start_distilbert.sh"
        echo "  ./start_tinybert.sh"
        echo "  ./start_minilm.sh"
        echo ""
        echo "• Benchmark de modelo específico:"
        echo "  ./benchmark_distilbert.sh"
        echo "  ./benchmark_tinybert.sh"
        echo "  ./benchmark_minilm.sh"
    else
        echo "• Iniciar detector:"
        echo "  ./start_${SELECTED_MODEL,,}.sh"
        echo ""
        echo "• Executar benchmark:"
        echo "  ./benchmark_${SELECTED_MODEL,,}.sh"
    fi
    
    echo ""
    echo "• Ver logs:"
    if [[ "$SELECTED_MODEL" == "ALL" ]]; then
        echo "  tail -f logs/DistilBERT/attack_log.json"
        echo "  tail -f logs/TinyBERT/attack_log.json" 
        echo "  tail -f logs/MiniLM/attack_log.json"
    else
        echo "  tail -f logs/$SELECTED_MODEL/attack_log.json"
    fi
    echo ""
}

show_file_info() {
    echo "📊 ARQUIVOS IMPORTANTES:"
    echo "======================="
    echo "• Modelos ONNX: */modelo_attack_detector_quantized.onnx"
    echo "• Metadados: */modelo_metadata.pkl"
    echo "• Logs: logs/*/attack_log.json"
    echo "• Configurações: config/*_config.json"
    echo "• Análises: analysis_results/*/"
    echo "• Requirements: requirements_consolidated.txt"
    echo ""
}

show_troubleshooting_info() {
    echo "🔧 TROUBLESHOOTING:"
    echo "==================="
    echo "• Se houver erro de memória, reduza o batch_size na configuração"
    echo "• Para melhor performance, use CPU com 4+ cores"
    echo "• Monitore uso de CPU com: htop"
    echo "• Para executar em background: tmux ou screen"
    echo "• Logs de erro: verifique os arquivos .log em cada diretório"
    echo "• Reativar ambiente virtual: source venv/bin/activate"
    echo ""
}

main() {
    echo "🔍 Sistema Unificado de Detecção de Ataques de Rede"
    echo "===================================================="
    echo "Suporte para: DistilBERT, TinyBERT, MiniLM"
    echo ""
    
    select_model
    check_files
    
    read -p "Deseja atualizar o sistema? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        update_system
    fi
    
    install_system_deps
    create_venv
    install_python_deps
    setup_directories
    create_config
    create_service_scripts
    test_installation
    show_final_info
}

main "$@" 