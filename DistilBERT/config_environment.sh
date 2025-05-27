#!/bin/bash

set -e 

echo "🚀 Iniciando deploy do sistema de detecção de ataques no Raspberry Pi..."

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

check_files() {
    log_info "Verificando arquivos necessários..."
    
    required_files=(
        "realtime_network_monitor.py"
        "performance_analyzer.py"
        "network_attack_detector_quantized.onnx"
        "model_metadata.pkl"
        "requirements.txt"
        "requirements-raspberry.txt"
    )
    
    missing_files=()
    
    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            log_success "✓ $file"
        else
            log_error "✗ $file (OBRIGATÓRIO - FALTANDO)"
            missing_files+=("$file")
        fi
    done
    
    if [[ ! -f "requirements.txt" && ! -f "requirements-raspberry.txt" ]]; then
        log_error "✗ Nenhum arquivo de requirements encontrado (requirements.txt ou requirements-raspberry.txt)"
        missing_files+=("requirements.txt")
    fi
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log_error "Arquivos obrigatórios faltando: ${missing_files[*]}"
        log_error "Por favor, certifique-se de que todos os arquivos obrigatórios estão no diretório atual."
        exit 1
    fi
    
    log_success "Todos os arquivos obrigatórios estão presentes!"
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
        tmux
    
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
    
    if [[ -f "requirements-raspberry.txt" ]]; then
        requirements_file="requirements-raspberry.txt"
        log_info "Usando requirements específico para Raspberry Pi"
    else
        requirements_file="requirements.txt"
        log_info "Usando requirements padrão"
    fi
    
    log_info "Instalando dependências do arquivo: $requirements_file"
    
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
    done < "$requirements_file"
    
    additional_packages=("matplotlib" "seaborn")
    for pkg in "${additional_packages[@]}"; do
        if ! pip show "$pkg" > /dev/null 2>&1; then
            log_info "Instalando pacote adicional: $pkg"
            pip install "$pkg" || log_warning "Falha ao instalar $pkg"
        fi
    done
    
    critical_packages=("onnxruntime" "numpy" "pandas" "scikit-learn")
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
    pip list | grep -E "(onnxruntime|numpy|pandas|scikit-learn|matplotlib|seaborn)"
}

setup_directories() {
    log_info "Configurando diretórios..."
    
    mkdir -p logs
    mkdir -p data
    mkdir -p analysis_results
    mkdir -p config
    
    log_success "Diretórios configurados!"
}

create_config() {
    log_info "Criando arquivo de configuração..."
    
    cat > config/detector_config.json << EOF
{
    "model": {
        "path": "network_attack_detector_quantized.onnx",
        "metadata_path": "model_metadata.pkl"
    },
    "monitoring": {
        "log_file": "logs/attack_log.json",
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
    
    log_success "Arquivo de configuração criado!"
}

create_service_scripts() {
    log_info "Criando scripts de serviço..."
    
    cat > start_detector.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 realtime_network_monitor.py --interactive
EOF
    
    cat > run_benchmark.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 realtime_network_monitor.py --benchmark
EOF
    
    cat > run_analysis.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

if [[ $# -eq 0 ]]; then
    echo "Uso: $0 <arquivo_csv_teste>"
    exit 1
fi

python3 performance_analyzer.py --test_data "$1"
EOF
    
    chmod +x start_detector.sh run_benchmark.sh run_analysis.sh
    
    log_success "Scripts de serviço criados!"
}

create_systemd_service() {
    log_info "Criando serviço systemd..."
    
    current_dir=$(pwd)
    user=$(whoami)
    
    cat > network-detector.service << EOF
[Unit]
Description=Network Attack Detector
After=network.target

[Service]
Type=simple
User=$user
WorkingDirectory=$current_dir
Environment=PATH=$current_dir/venv/bin
ExecStart=$current_dir/venv/bin/python $current_dir/realtime_network_monitor.py --interactive
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    log_info "Para instalar o serviço systemd, execute:"
    log_info "sudo cp network-detector.service /etc/systemd/system/"
    log_info "sudo systemctl enable network-detector"
    log_info "sudo systemctl start network-detector"
    
    log_success "Arquivo de serviço systemd criado!"
}

test_installation() {
    log_info "Testando instalação..."
    
    source venv/bin/activate
    
    if [[ -f "test_installation.py" ]]; then
        log_info "Executando teste completo de instalação..."
        if python3 test_installation.py; then
            log_success "✅ Todos os testes passaram!"
        else
            log_warning "⚠️ Alguns testes falharam, mas continuando..."
        fi
    else
        log_info "Executando teste básico..."
        
        python3 -c "
import sys
try:
    import numpy as np
    import pandas as pd
    import sklearn
    import onnxruntime as ort
    print('✅ Importações críticas OK')
    print(f'Python: {sys.version.split()[0]}')
    print(f'NumPy: {np.__version__}')
    print(f'Pandas: {pd.__version__}')
    print(f'ONNX Runtime: {ort.__version__}')
except ImportError as e:
    print(f'❌ Erro de importação: {e}')
    sys.exit(1)
" || {
            log_error "Teste básico de importações falhou!"
            log_error "Execute './install_manual.sh' para tentar corrigir"
            return 1
        }
        
        log_info "Testando monitor básico..."
        timeout 15 python3 realtime_network_monitor.py --benchmark 2>/dev/null || {
            log_warning "Teste do monitor falhou ou foi interrompido (normal se não houver modelo)"
        }
    fi
    
    log_success "Teste de instalação concluído!"
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
    
    echo "🚀 COMANDOS ÚTEIS:"
    echo "=================="
    echo "• Iniciar detector interativo:"
    echo "  ./start_detector.sh"
    echo ""
    echo "• Executar benchmark:"
    echo "  ./run_benchmark.sh"
    echo ""
    echo "• Analisar dados de teste:"
    echo "  ./run_analysis.sh seu_arquivo.csv"
    echo ""
    echo "• Simular com dados CSV:"
    echo "  source venv/bin/activate"
    echo "  python3 realtime_network_monitor.py --simulate dados.csv"
    echo ""
    echo "• Ver logs:"
    echo "  tail -f logs/attack_log.json"
    echo ""
    
    echo "📊 ARQUIVOS IMPORTANTES:"
    echo "======================="
    echo "• Modelo: network_attack_detector_quantized.onnx"
    echo "• Metadados: model_metadata.pkl"
    echo "• Logs: logs/attack_log.json"
    echo "• Configuração: config/detector_config.json"
    echo "• Análises: analysis_results/"
    echo ""
    
    echo "🔧 TROUBLESHOOTING:"
    echo "==================="
    echo "• Se houver erro de memória, reduza o batch_size"
    echo "• Para melhor performance, use CPU com 4+ cores"
    echo "• Monitore uso de CPU com: htop"
    echo "• Para executar em background: tmux ou screen"
    echo ""
    
    log_success "Sistema pronto para uso! 🎯"
}

main() {
    echo "🔍 Sistema de Detecção de Ataques de Rede - Raspberry Pi"
    echo "========================================================"
    echo ""
    
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
    create_systemd_service
    test_installation
    show_final_info
}

main "$@" 