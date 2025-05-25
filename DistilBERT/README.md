# Sistema de Detecção de Ataques de Rede - Raspberry Pi

Sistema de detecção de ataques DDoS em tempo real usando DistilBERT otimizado para IoT.

## 🚀 Instalação Rápida

### Opção 1: Script Automático (Recomendado)
```bash
# Baixar todos os arquivos para um diretório
# Executar o script de deploy
chmod +x deploy_raspberry_pi.sh
./deploy_raspberry_pi.sh
```

### Opção 2: Instalação Manual (Se houver problemas)
```bash
# Use o script manual para resolver dependências
chmod +x install_manual.sh
./install_manual.sh
```

### Opção 3: Instalação Passo a Passo
```bash
# 1. Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# 2. Atualizar pip
pip install --upgrade pip

# 3. Instalar dependências (versões atualizadas)
pip install -r requirements-raspberry.txt

# 4. Testar instalação
python3 test_installation.py
```

## 📋 Arquivos Necessários

### Obrigatórios
- `realtime_network_monitor.py` - Monitor em tempo real
- `performance_analyzer.py` - Analisador de performance
- `requirements.txt` ou `requirements-raspberry.txt` - Dependências

### Opcionais (para funcionamento completo)
- `network_attack_detector_quantized.onnx` - Modelo treinado
- `model_metadata.pkl` - Metadados do modelo
- Arquivos CSV com dados de rede para teste

## 🔧 Solução de Problemas

### Erro de Sintaxe Python
```bash
# Se aparecer: "SyntaxError: unterminated string literal"
python test_syntax.py  # Verificar sintaxe de todos os arquivos
```

### Erro de Versão ONNX Runtime
```bash
# Se aparecer: "No matching distribution found for onnxruntime==1.15.1"
pip install onnxruntime>=1.17.0
```

### Problemas de Memória
```bash
# Aumentar swap no Raspberry Pi
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Alterar: CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Teste Rápido
```bash
# Verificar se tudo está funcionando
source venv/bin/activate
python3 test_installation.py
```

## 🎯 Uso do Sistema

### 1. Teste de Performance
```bash
source venv/bin/activate
python3 realtime_network_monitor.py --benchmark
```

### 2. Monitoramento Interativo
```bash
./start_detector.sh
```

### 3. Simulação com Dados CSV
```bash
source venv/bin/activate
python3 realtime_network_monitor.py --simulate seus_dados.csv
```

### 4. Análise de Performance
```bash
python3 performance_analyzer.py --test_data seus_dados.csv
```

## 📊 Estrutura de Arquivos

```
projeto/
├── deploy_raspberry_pi.sh          # Script de instalação automática
├── install_manual.sh               # Script de instalação manual
├── test_installation.py            # Teste de verificação
├── requirements.txt                # Dependências padrão
├── requirements-raspberry.txt       # Dependências para Raspberry Pi
├── realtime_network_monitor.py     # Monitor principal
├── performance_analyzer.py         # Analisador de métricas
├── TROUBLESHOOTING.md              # Guia de solução de problemas
├── network_attack_detector_quantized.onnx  # Modelo (após treinamento)
├── model_metadata.pkl              # Metadados (após treinamento)
├── logs/                           # Logs do sistema
├── data/                           # Dados de entrada
├── analysis_results/               # Resultados de análise
└── config/                         # Configurações
```

## 🔍 Verificação da Instalação

### Comandos de Diagnóstico
```bash
# Verificar versões
python3 --version
pip list | grep -E "(onnx|numpy|pandas|sklearn)"

# Testar importações
python3 -c "import onnxruntime; print('ONNX OK')"

# Verificar sistema
free -h  # Memória
df -h    # Espaço em disco
```

### Logs Importantes
- `logs/attack_log.json` - Detecções em tempo real
- `analysis_results/` - Relatórios de análise
- `test_plot.png` - Teste de visualização

## ⚡ Performance Esperada

### Raspberry Pi 4
- **Throughput**: 100+ amostras/segundo
- **Latência**: <50ms por predição
- **Memória**: <100MB
- **Modelo**: ~60MB (quantizado)

### Raspberry Pi 3
- **Throughput**: 50+ amostras/segundo
- **Latência**: <100ms por predição
- **Configuração**: Reduzir batch_size para 8-16

## 🛠️ Configuração Avançada

### Arquivo de Configuração
```json
{
    "model": {
        "path": "network_attack_detector_quantized.onnx",
        "metadata_path": "model_metadata.pkl"
    },
    "monitoring": {
        "batch_size": 32,
        "alert_threshold": 0.8
    }
}
```

### Serviço Systemd
```bash
# Instalar como serviço
sudo cp network-detector.service /etc/systemd/system/
sudo systemctl enable network-detector
sudo systemctl start network-detector
```

## 📈 Monitoramento

### Logs em Tempo Real
```bash
tail -f logs/attack_log.json
```

### Monitoramento de Recursos
```bash
htop                    # CPU e memória
vcgencmd measure_temp   # Temperatura (Raspberry Pi)
```

### Execução em Background
```bash
tmux new-session -d -s detector
tmux send-keys -t detector "source venv/bin/activate" Enter
tmux send-keys -t detector "python3 realtime_network_monitor.py --interactive" Enter
```

## 🆘 Suporte

### Se algo não funcionar:
1. **Consulte**: `TROUBLESHOOTING.md`
2. **Execute**: `python3 test_installation.py`
3. **Tente**: `./install_manual.sh`
4. **Verifique**: Versões das dependências

### Informações para Suporte
Ao reportar problemas, inclua:
- Saída de `python3 --version`
- Saída de `uname -a`
- Erro completo
- Saída de `pip list`

## 🎯 Próximos Passos

1. **Treinar Modelo**: Use o notebook no Google Colab com seus dados CSV
2. **Transferir Arquivos**: Baixe `.onnx` e `.pkl` para o Raspberry Pi
3. **Executar Deploy**: Use o script de instalação
4. **Monitorar**: Configure alertas e logs
5. **Otimizar**: Ajuste parâmetros conforme necessário

## 📝 Notas Importantes

- **Python 3.8+** requerido
- **4GB+ RAM** recomendado para Raspberry Pi
- **Cartão SD Classe 10** ou SSD para melhor performance
- **Conexão de rede** estável para monitoramento
- **Dissipador de calor** recomendado para uso contínuo
