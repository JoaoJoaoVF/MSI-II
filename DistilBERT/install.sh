#!/bin/bash
# Script de instalação para Raspberry Pi

echo "Instalando dependências para detecção de ataques de rede..."

# Atualizar sistema
sudo apt update
sudo apt upgrade -y

# Instalar Python e pip
sudo apt install python3 python3-pip -y

# Instalar dependências Python
pip3 install -r requirements.txt

# Criar diretório de logs
mkdir -p logs

echo "Instalação concluída!"
echo "Para testar: python3 realtime_network_monitor.py --benchmark"
