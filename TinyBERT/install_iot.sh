#!/bin/bash
# Script de instalação TinyBERT para IoT extremo

echo "🚀 Instalando TinyBERT para IoT..."

# Configurações de sistema para IoT
echo "Configurando sistema..."
export OMP_NUM_THREADS=1
export ONNX_DISABLE_STATIC_ANALYSIS=1
ulimit -v 300000  # Limitar memória virtual a 300MB

# Instalar dependências mínimas
pip3 install --no-cache-dir -r requirements_iot.txt

# Criar diretório de logs compactos
mkdir -p logs

echo "✅ TinyBERT IoT instalado!"
echo "Teste: python3 tinybert_network_monitor.py --benchmark --samples 100"
