#!/bin/bash
# Script de instalação MiniLM para Raspberry Pi

echo "🚀 Instalando MiniLM para Raspberry Pi..."

# Atualizar sistema
sudo apt update
sudo apt upgrade -y

# Instalar Python e pip
sudo apt install python3 python3-pip -y

# Configurações otimizadas para MiniLM
export OMP_NUM_THREADS=2
export ONNX_DISABLE_STATIC_ANALYSIS=1

# Instalar dependências Python
pip3 install -r requirements.txt

# Criar diretório de logs
mkdir -p logs

echo "✅ MiniLM instalado com sucesso!"
echo "Para testar: python3 minilm_network_monitor.py --benchmark"
