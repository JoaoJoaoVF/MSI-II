#!/bin/bash
# Script de Instalação Manual - Fallback para problemas de dependências
# Use este script se o deploy automático falhar

echo "🔧 Instalação Manual - Sistema de Detecção de Ataques"
echo "===================================================="

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Este script instala as dependências uma por uma para resolver problemas.${NC}"
echo ""

# Criar ambiente virtual
echo "1. Criando ambiente virtual..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo ""
echo "2. Instalando dependências básicas..."

# Instalar dependências uma por uma
packages=(
    "numpy"
    "pandas" 
    "scikit-learn"
    "matplotlib"
    "seaborn"
    "onnxruntime"
)

for package in "${packages[@]}"; do
    echo -e "${YELLOW}Instalando $package...${NC}"
    if pip install "$package"; then
        echo -e "${GREEN}✓ $package instalado com sucesso${NC}"
    else
        echo -e "${RED}✗ Falha ao instalar $package${NC}"
        echo "Tentando versão mais recente..."
        pip install "$package" --upgrade --no-deps || echo -e "${RED}Falha crítica em $package${NC}"
    fi
    echo ""
done

echo ""
echo "3. Verificando instalações..."

# Verificar se os pacotes críticos foram instalados
critical_check() {
    local package=$1
    if python3 -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✓ $package OK${NC}"
        return 0
    else
        echo -e "${RED}✗ $package FALHOU${NC}"
        return 1
    fi
}

echo "Verificando pacotes críticos:"
critical_check "numpy"
critical_check "pandas" 
critical_check "sklearn"
critical_check "onnxruntime"

echo ""
echo "4. Testando importações..."

# Teste básico de importação
python3 -c "
try:
    import numpy as np
    import pandas as pd
    import sklearn
    import onnxruntime as ort
    print('✓ Todas as importações críticas funcionaram!')
except ImportError as e:
    print(f'✗ Erro de importação: {e}')
"

echo ""
echo "5. Criando diretórios..."
mkdir -p logs data analysis_results config

echo ""
echo -e "${GREEN}Instalação manual concluída!${NC}"
echo ""
echo "Para testar o sistema:"
echo "1. source venv/bin/activate"
echo "2. python3 realtime_network_monitor.py --benchmark"
echo ""
echo "Se ainda houver problemas, tente:"
echo "- pip install --upgrade pip setuptools wheel"
echo "- pip install onnxruntime --no-deps"
echo "- Verificar se está usando Python 3.8+ com: python3 --version" 