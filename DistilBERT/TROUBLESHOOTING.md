# Guia de Solução de Problemas - Sistema de Detecção de Ataques

## 🚨 Problemas Comuns e Soluções

### 1. Erro de Versão do ONNX Runtime

**Problema**: `ERROR: No matching distribution found for onnxruntime==1.15.1`

**Solução**:
```bash
# Use versões mais recentes
pip install onnxruntime>=1.17.0

# Ou instale a versão mais recente disponível
pip install onnxruntime --upgrade
```

### 2. Problemas de Dependências no Raspberry Pi

**Problema**: Falhas na instalação de pacotes

**Soluções**:

#### Opção A: Script Manual
```bash
chmod +x install_manual.sh
./install_manual.sh
```

#### Opção B: Instalação Individual
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Instalar um por vez
pip install numpy
pip install pandas
pip install scikit-learn
pip install onnxruntime
pip install matplotlib
pip install seaborn
```

#### Opção C: Usar requirements atualizado
```bash
# Use o arquivo requirements-raspberry.txt
pip install -r requirements-raspberry.txt
```

### 3. Erro de Memória no Raspberry Pi

**Problema**: `MemoryError` ou `Killed` durante instalação

**Soluções**:
```bash
# Aumentar swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Alterar CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Instalar com cache limitado
pip install --no-cache-dir onnxruntime
```

### 4. Problemas de Arquitetura ARM

**Problema**: Pacotes não compatíveis com ARM

**Soluções**:
```bash
# Para numpy/scipy no ARM
sudo apt install python3-numpy python3-scipy python3-matplotlib
pip install --no-deps scikit-learn

# Para onnxruntime no ARM
pip install onnxruntime --extra-index-url https://pypi.org/simple/
```

### 5. Erro de Importação

**Problema**: `ImportError` ao executar scripts

**Diagnóstico**:
```bash
source venv/bin/activate
python3 -c "
import sys
print('Python version:', sys.version)
try:
    import numpy; print('✓ numpy')
    import pandas; print('✓ pandas')
    import sklearn; print('✓ sklearn')
    import onnxruntime; print('✓ onnxruntime')
except ImportError as e:
    print('✗ Import error:', e)
"
```

### 6. Modelo ONNX não encontrado

**Problema**: `FileNotFoundError: network_attack_detector_quantized.onnx`

**Soluções**:
1. Certifique-se de ter executado o notebook de treinamento
2. Baixe o modelo do Google Colab
3. Coloque o arquivo no diretório correto
4. Verifique permissões: `ls -la *.onnx`

### 7. Performance Baixa

**Problema**: Inferência muito lenta

**Otimizações**:
```bash
# Verificar CPU
cat /proc/cpuinfo | grep "model name"

# Monitorar recursos
htop

# Reduzir batch size no código
# Editar realtime_network_monitor.py
# batch_size = 16  # em vez de 32
```

### 8. Erro de Permissões

**Problema**: `Permission denied`

**Soluções**:
```bash
# Tornar scripts executáveis
chmod +x *.sh

# Verificar proprietário
ls -la

# Corrigir permissões se necessário
sudo chown -R $USER:$USER .
```

### 9. Avisos do StandardScaler

**Problema**: `UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names`

**Causa**: O scaler foi treinado com DataFrame (com nomes de features) mas está recebendo array NumPy (sem nomes)

**Soluções**:
```bash
# Já corrigido no código, mas se persistir:

# Opção 1: Executar script de demonstração
python fix_feature_warnings.py

# Opção 2: Suprimir avisos temporariamente
export PYTHONWARNINGS='ignore::UserWarning'
python3 realtime_network_monitor.py --simulate dados.csv

# Opção 3: Verificar se o problema foi corrigido
python3 -c "
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
print('Avisos suprimidos com sucesso')
"
```

## 🔧 Comandos de Diagnóstico

### Verificar Sistema
```bash
# Versão do Python
python3 --version

# Arquitetura
uname -a

# Memória disponível
free -h

# Espaço em disco
df -h

# Temperatura (Raspberry Pi)
vcgencmd measure_temp
```

### Verificar Instalação
```bash
# Ativar ambiente
source venv/bin/activate

# Listar pacotes instalados
pip list

# Verificar pacotes críticos
pip show onnxruntime numpy pandas scikit-learn

# Testar importações
python3 -c "import onnxruntime; print('ONNX Runtime version:', onnxruntime.__version__)"
```

### Testar Sistema
```bash
# Teste básico
python3 realtime_network_monitor.py --benchmark

# Teste com dados
python3 realtime_network_monitor.py --simulate dados.csv

# Análise de performance
python3 performance_analyzer.py --test_data dados.csv
```

## 📞 Suporte Adicional

### Logs Úteis
- `logs/attack_log.json` - Logs de detecção
- `analysis_results/` - Relatórios de análise
- `result-*.txt` - Arquivos de resultado da análise
- `pip list` - Pacotes instalados

### Informações para Suporte
Ao reportar problemas, inclua:
1. Versão do Python: `python3 --version`
2. Sistema operacional: `uname -a`
3. Arquitetura: `arch`
4. Memória: `free -h`
5. Erro completo
6. Comando que causou o erro

### Reinstalação Completa
```bash
# Remover ambiente virtual
rm -rf venv

# Limpar cache pip
pip cache purge

# Reinstalar do zero
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
./install_manual.sh
```

## 🎯 Dicas de Performance

### Para Raspberry Pi 4
- Use cartão SD classe 10 ou SSD
- Ative overclock moderado
- Use dissipador de calor
- Monitore temperatura

### Para Raspberry Pi 3
- Reduza batch_size para 8-16
- Use swap de 1GB+
- Considere modelo ainda mais leve
- Execute apenas detecção essencial

### Otimizações Gerais
- Execute em background com `tmux`
- Use `nice` para prioridade baixa
- Monitore com `htop`
- Configure logrotate para logs 

### Arquivos de Resultado
O sistema agora gera automaticamente arquivos de resultado:
- **Formato**: `result-nome_do_csv.txt`
- **Conteúdo**: Estatísticas completas, tipos de ataques, detalhes de cada detecção
- **Localização**: Diretório atual de execução
- **Exemplo**: Para `network_data.csv` → `result-network_data.txt` 