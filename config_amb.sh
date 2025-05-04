#!/usr/bin/env bash
# setup_ml.sh – Automatiza a instalação do stack de Machine Learning
# Testado em Ubuntu Server 24.04 LTS (kernel 6.8) numa VM VirtualBox/KVM.
# Executar com:  bash setup_ml.sh
# Após o término, ative o ambiente com: source $HOME/env-ml/bin/activate

set -euo pipefail

############################################
# 1. Atualização do sistema
############################################
echo "[1/6] Atualizando repositórios e pacotes..."
sudo apt update && sudo apt -y upgrade

############################################
# 2. Instala dependências de compilação e utilitários básicos
############################################
echo "[2/6] Instalando dependências de sistema..."
sudo apt install -y build-essential git curl wget python3 python3-venv python3-pip

############################################
# 3. Cria e configura ambiente Python isolado (venv)
############################################
ENV_DIR="$HOME/env-ml"
if [ ! -d "$ENV_DIR" ]; then
  echo "[3/6] Criando ambiente virtual em $ENV_DIR ..."
  python3 -m venv "$ENV_DIR"
fi

# Ativa o venv para o restante do script
source "$ENV_DIR/bin/activate"

# Garante ferramentas de empacotamento atualizadas
pip install -U pip wheel setuptools

############################################
# 4. Instala PyTorch CPU‑only (2.4+) e bibliotecas correlatas
############################################
echo "[4/6] Instalando PyTorch CPU‑only..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

############################################
# 5. Instala stack Hugging Face + ONNX + quantização
############################################
echo "[5/6] Instalando Transformers, Optimum, ONNX Runtime, bitsandbytes e utilitários..."
# Transformers (modelos e tokenizers), Optimum (exportação/quantização), ONNX & ferramentas
pip install \
  transformers \
  "optimum[onnxruntime]" \
  onnx \
  onnxruntime \
  onnxruntime-tools \
  bitsandbytes==0.45.5 \
  datasets \
  evaluate

############################################
# 6. Limpeza e mensagem final
############################################
echo "[6/6] Instalação concluída com sucesso!"
echo "Para usar, ative o ambiente com: source $ENV_DIR/bin/activate"
echo "Exemplo rápido de teste:"
echo "python - <<'PY'\nfrom transformers import AutoModel; AutoModel.from_pretrained('distilbert/distilbert-base-uncased'); print('Modelo carregado!')\nPY"
