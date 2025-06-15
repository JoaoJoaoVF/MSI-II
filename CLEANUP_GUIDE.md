# 🧹 Guia de Limpeza dos Modelos

## 📋 Arquivos ESSENCIAIS (NÃO REMOVER)

Baseado na análise dos scripts `setup_environment.sh` e `run_all_monitors.sh`, estes são os arquivos que **DEVEM SER MANTIDOS** em cada pasta:

### 📁 **DistilBERT/** - Arquivos obrigatórios:

```
✅ MANTER:
├── realtime_network_monitor.py          # Script principal de monitoramento
├── performance_analyzer.py              # Análise de performance
├── network_attack_detector_quantized.onnx  # Modelo ONNX quantizado
├── model_metadata.pkl                   # Metadados do modelo
├── requirements.txt                     # Dependências Python
└── README.md                           # Documentação (opcional)
```

### 📁 **MiniLM/** - Arquivos obrigatórios:

```
✅ MANTER:
├── realtime_network_monitor.py          # Script principal de monitoramento
├── minilm_network_monitor.py           # Monitor específico do MiniLM
├── minilm_attack_detector_quantized.onnx  # Modelo ONNX quantizado
├── minilm_metadata.pkl                 # Metadados do modelo
├── requirements.txt                     # Dependências Python
└── README.md                           # Documentação (opcional)
```

### 📁 **TinyBERT/** - Arquivos obrigatórios:

```
✅ MANTER:
├── realtime_network_monitor.py          # Script principal de monitoramento
├── tinybert_network_monitor.py         # Monitor específico do TinyBERT
├── tinybert_attack_detector_quantized.onnx  # Modelo ONNX quantizado
├── tinybert_metadata.pkl               # Metadados do modelo
├── requirements.txt                     # Dependências Python
└── README.md                           # Documentação (opcional)
```

---

## 🗑️ Arquivos REMOVÍVEIS (podem ser apagados)

### Dados duplicados:

- `part-*.csv` - Arquivos CSV duplicados da pasta `data/`

### Resultados de execuções anteriores:

- `result-*.txt` - Arquivos de resultado de simulações
- `attack_log.json` - Logs de ataques antigos
- `*_attack_log.json` - Logs específicos de cada modelo

### Notebooks e scripts de desenvolvimento:

- `*_optimization.ipynb` - Notebooks Jupyter de otimização
- `test_*.py` - Scripts de teste
- `run_*.sh` - Scripts redundantes de execução
- `start_detector.sh` - Scripts específicos redundantes

### Configurações específicas:

- `network-detector.service` - Configuração de serviço systemd
- `config_environment.sh` - Configurações específicas de ambiente
- `device_configs.sh` - Configurações de dispositivo
- `install_*.sh` - Scripts de instalação específicos

### Dependências específicas:

- `requirements-raspberry.txt` - Requirements para Raspberry Pi
- `requirements_iot.txt` - Requirements para IoT
- `requirements_*.txt` (exceto requirements.txt principal)

### Arquivos temporários:

- `*.log` - Arquivos de log antigos
- `*.tmp` - Arquivos temporários
- `*.bak` - Arquivos de backup

---

## 🚀 Como usar o script de limpeza

### 1. Análise primeiro (recomendado):

```bash
# Analisar sem remover nada
./cleanup_models.sh --analyze
```

### 2. Limpeza interativa:

```bash
# Modo interativo com menu
./cleanup_models.sh
```

### 3. Limpeza automática:

```bash
# Limpar todos os modelos automaticamente
./cleanup_models.sh --clean-all
```

### 4. Backup antes da limpeza:

```bash
# Criar backup de todos os modelos
./cleanup_models.sh --backup
```

---

## ⚠️ IMPORTANTE

### Antes de executar a limpeza:

1. **Faça backup** dos arquivos importantes
2. **Verifique** se você tem os arquivos essenciais
3. **Teste** se o sistema ainda funciona após a limpeza

### Verificação pós-limpeza:

```bash
# Verificar se todos os arquivos essenciais estão presentes
./cleanup_models.sh --verify
```

### Se algo der errado:

1. Os backups ficam salvos como `backup_[modelo]_[data]`
2. Você pode restaurar copiando os arquivos de volta
3. Os scripts `setup_environment.sh` podem recriar configurações básicas

---

## 📊 Estimativa de espaço economizado

Com base na estrutura atual, você pode economizar:

- **DistilBERT**: ~50-100MB (resultados + duplicatas)
- **MiniLM**: ~30-50MB (resultados + duplicatas)
- **TinyBERT**: ~20-40MB (resultados + duplicatas)
- **Total**: ~100-200MB de espaço

---

## 🔧 Scripts que dependem dos arquivos essenciais

### `setup_environment.sh` verifica:

- Scripts Python principais (`realtime_network_monitor.py`)
- Modelos ONNX (`*_quantized.onnx`)
- Metadados (`*_metadata.pkl`)
- Dependências (`requirements.txt`)

### `run_all_monitors.sh` executa:

- `realtime_network_monitor.py` (script principal)
- Scripts específicos (`*_network_monitor.py`)

### Nunca remova:

- Arquivos `.onnx` (modelos de machine learning)
- Arquivos `.pkl` (metadados e configurações)
- Scripts `.py` principais
- `requirements.txt` principal

---

## 💡 Dicas

1. **Sempre analise primeiro**: Use `--analyze` antes de remover
2. **Faça backup**: Especialmente na primeira vez
3. **Teste após limpeza**: Execute os scripts para verificar se funcionam
4. **Monitore espaço**: Use `./cleanup_models.sh` opção 7 para ver uso de disco
5. **Logs são recriados**: Os `attack_log.json` são recriados automaticamente
