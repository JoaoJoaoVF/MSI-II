# DistilBERT otimizado para IoT

## 🚀 Instalação

```bash
chmod +x deploy.sh
./deploy.sh
```

ou

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip

pip install -r requirements-raspberry.txt

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

### 3. Simulação com Dados CSV (com arquivo de resultado)

```bash
source venv/bin/activate
# Exemplo: se o arquivo for "network_data.csv", será criado "result-network_data.txt"
python3 realtime_network_monitor.py --simulate network_data.csv

```

### 4. Análise de Performance

```bash
python3 performance_analyzer.py --test_data seus_dados.csv
```

## 📊 Estrutura de Arquivos

```
projeto/
├── config_environment.sh                   # Script de instalação automática
├── requirements.txt                        # Dependências padrão
├── requirements-raspberry.txt              # Dependências para Raspberry Pi
├── realtime_network_monitor.py             # Monitor principal
├── performance_analyzer.py                 # Analisador de métricas
├── network_attack_detector_quantized.onnx  # Modelo (após treinamento)
├── model_metadata.pkl                      # Metadados (após treinamento)
├── logs/                                   # Logs do sistema
├── analysis_results/                       # Resultados de análise
└── config/                                 # Configurações
```
