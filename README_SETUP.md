# Sistema Unificado de Detecção de Ataques de Rede

Este projeto contém três modelos de machine learning otimizados para detecção de ataques de rede: **DistilBERT**, **TinyBERT** e **MiniLM**.

## 📁 Estrutura do Projeto

```
MSI-II/
├── setup_environment.sh          # Script de configuração unificado
├── DistilBERT/                   # Modelo DistilBERT
│   ├── realtime_network_monitor.py    # Monitor unificado DistilBERT
│   ├── performance_analyzer.py
│   ├── network_attack_detector_quantized.onnx
│   ├── model_metadata.pkl
│   └── requirements.txt
├── TinyBERT/                     # Modelo TinyBERT  
│   ├── realtime_network_monitor.py    # Monitor unificado TinyBERT (IoT)
│   ├── tinybert_network_monitor.py    # Monitor original TinyBERT
│   ├── tinybert_attack_detector_quantized.onnx
│   ├── tinybert_metadata.pkl
│   └── requirements.txt
├── MiniLM/                       # Modelo MiniLM
│   ├── realtime_network_monitor.py    # Monitor unificado MiniLM (Workstation)
│   ├── minilm_network_monitor.py      # Monitor original MiniLM
│   ├── minilm_attack_detector_quantized.onnx
│   ├── minilm_metadata.pkl
│   └── requirements.txt
└── README_SETUP.md               # Este arquivo
```

## 🚀 Instalação e Configuração

### Pré-requisitos

- **Sistema Operacional**: Linux (Ubuntu/Debian recomendado) ou Raspberry Pi
- **Python**: 3.8 ou superior
- **Memória**: Mínimo 4GB RAM (8GB recomendado)
- **Espaço em Disco**: Mínimo 2GB livre

### Instalação Rápida

1. **Clone ou baixe o projeto** para o diretório desejado

2. **Execute o script de configuração**:
   ```bash
   chmod +x setup_environment.sh
   ./setup_environment.sh
   ```

3. **Selecione o modelo desejado**:
   - `1` - DistilBERT (melhor precisão)
   - `2` - TinyBERT (menor consumo de recursos)
   - `3` - MiniLM (equilíbrio entre precisão e performance)
   - `4` - Todos os modelos (configuração completa)

### O que o Script Faz

O `setup_environment.sh` realiza automaticamente:

✅ **Verificação de Arquivos**: Confirma se todos os arquivos necessários estão presentes  
✅ **Atualização do Sistema**: Atualiza os pacotes do sistema (opcional)  
✅ **Instalação de Dependências**: Instala Python, bibliotecas do sistema e dependências  
✅ **Ambiente Virtual**: Cria e configura um ambiente Python isolado  
✅ **Configuração**: Cria arquivos de configuração personalizados  
✅ **Scripts de Controle**: Gera scripts para iniciar, parar e testar os modelos  
✅ **Teste de Instalação**: Verifica se tudo foi instalado corretamente  

## 🎯 Uso do Sistema

### Comandos Principais

#### Para um modelo específico:
```bash
# Iniciar detector
./start_distilbert.sh
./start_tinybert.sh  
./start_minilm.sh

# Executar benchmark
./benchmark_distilbert.sh
./benchmark_tinybert.sh
./benchmark_minilm.sh

# Simular com dados CSV
./simulate_distilbert.sh data/network_data.csv
./simulate_tinybert.sh data/network_data.csv
./simulate_minilm.sh data/network_data.csv
```

#### Para todos os modelos (se instalação completa):
```bash
# Iniciar todos os modelos
./start_all_models.sh

# Parar todos os modelos  
./stop_all_models.sh

# Benchmark de todos os modelos
./benchmark_all_models.sh
```

#### Uso direto dos monitores unificados:
```bash
# Entrar no diretório do modelo
cd DistilBERT  # ou TinyBERT/MiniLM

# Modo interativo
python3 realtime_network_monitor.py --interactive

# Benchmark
python3 realtime_network_monitor.py --benchmark

# Simulação com CSV (gera result-modelo-part-nome.txt)
python3 realtime_network_monitor.py --simulate ../data/network_data.csv --delay 0.1

# Arquivo de saída personalizado
python3 realtime_network_monitor.py --simulate data.csv --output custom-results.txt
```

**Nomes de Arquivos de Resultado:**
- **DistilBERT**: `result-distilbert-part-<nome_csv>.txt`
- **TinyBERT**: `result-tinybert-part-<nome_csv>.txt`  
- **MiniLM**: `result-minilm-part-<nome_csv>.txt`

### Monitoramento

```bash
# Ver logs em tempo real
tail -f logs/DistilBERT/attack_log.json
tail -f logs/TinyBERT/tinybert_attack_log.json
tail -f logs/MiniLM/minilm_attack_log.json

# Monitorar recursos do sistema
htop
```

## 📊 Características dos Modelos

| Modelo | Tamanho | RAM | Precisão | Velocidade | Otimização | Uso Recomendado |
|--------|---------|-----|----------|------------|------------|------------------|
| **DistilBERT** | ~818KB | 4-8GB | Alta | Média | Servidores | Análise detalhada, alta precisão |
| **TinyBERT** | ~195KB | 2-4GB | Média-Alta | Rápida | IoT/Edge | Dispositivos IoT, tempo real |
| **MiniLM** | ~692KB | 3-6GB | Alta | Média-Rápida | Workstation | Uso geral, balanceado |

### Optimizações Específicas por Modelo

#### 🔥 **TinyBERT (IoT/Edge)**
- Cache limitado (50 entradas)
- Warm-up mínimo (3 iterações)
- Limpeza agressiva de memória
- Alertas para latência > 5ms
- Buffer de queue menor (500)
- Threads limitadas para economia

#### ⚖️ **MiniLM (Workstation)**
- Cache moderado (200 entradas)
- Warm-up balanceado (10 iterações)
- Execução paralela otimizada
- Alertas para latência > 15ms
- Buffer de queue padrão (1000)
- Uso completo de CPU cores

#### 🎯 **DistilBERT (Servidor)**
- Interface consistente com outros modelos
- Lógica robusta de detecção de ataques
- Performance otimizada para análise detalhada
- Suporte a análise extensiva

## ⚙️ Configuração Avançada

### Arquivos de Configuração

Após a instalação, você encontrará:

```bash
config/
├── distilbert_config.json    # Configuração do DistilBERT
├── tinybert_config.json      # Configuração do TinyBERT  
└── minilm_config.json        # Configuração do MiniLM
```

### Parâmetros Ajustáveis

```json
{
    "monitoring": {
        "batch_size": 32,           # Tamanho do lote (reduzir se pouca RAM)
        "alert_threshold": 0.8      # Limite para alertas (0.0-1.0)
    },
    "performance": {
        "max_inference_time_ms": 100,  # Tempo máximo de inferência
        "target_throughput": 100       # Taxa de processamento alvo
    }
}
```

### Lógica de Detecção Unificada

Todos os modelos agora usam a mesma lógica robusta para classificação:

```python
# Verificar se é tráfego benigno primeiro
is_benign = predicted_class.lower() in ['benigntraffic', 'benign', 'normal']

if is_benign:
    # Tráfego benigno nunca é considerado ataque
    is_critical_attack = False
else:
    # Critérios para classificar como ataque crítico:
    is_critical_attack = (
        (high_confidence_attack and is_high_threat) or
        (high_confidence_attack and is_ddos) or
        (confidence > 0.9)
    )
```

**Categorias de Classificação:**
- **✅ TRÁFEGO NORMAL**: BenignTraffic, Benign, Normal (nunca considerados ataques)
- **🚨 ATAQUE CRÍTICO**: Ataques de alta confiança e alta ameaça
- **⚠️ ATIVIDADE SUSPEITA**: Alta confiança mas baixa ameaça
- **🔍 BAIXO RISCO**: Baixa confiança

**Parâmetros:**
- **Classes de baixa ameaça**: VulnerabilityScan, Recon-PingSweep, BrowserHijacking
- **Classes DDoS**: Automaticamente detectadas por padrão "DDoS" ou "DoS"
- **Threshold de confiança**: Configurável (padrão: 0.8)

## 🔧 Troubleshooting

### Problemas Comuns

#### ❌ Erro de Memória
```bash
# Solução: Reduzir batch_size na configuração
# Editar: config/modelo_config.json
"batch_size": 16  # ou menor

# Para TinyBERT em IoT extremamente limitado:
"batch_size": 8
```

#### ❌ Modelo não encontrado
```bash
# Verificar se os arquivos .onnx estão presentes
ls -la */realtime_network_monitor.py
ls -la */*_attack_detector_quantized.onnx
ls -la */*metadata.pkl
```

#### ❌ Dependências faltando
```bash
# Reativar ambiente virtual e reinstalar
source venv/bin/activate
pip install -r requirements_consolidated.txt
```

#### ❌ Performance baixa por modelo

**TinyBERT (IoT):**
```bash
# Verificar recursos limitados
free -h
# Considerar delay maior na simulação
python3 realtime_network_monitor.py --simulate data.csv --delay 0.2
```

**MiniLM (Workstation):**
```bash
# Verificar uso de CPU
htop
# Otimizar para paralelismo
# O MiniLM usa todos os cores disponíveis automaticamente
```

**DistilBERT (Servidor):**
```bash
# Verificar se há recursos suficientes
# Considerar usar MiniLM se recursos limitados
```

### Logs de Debug

```bash
# Logs específicos por modelo
tail -f logs/DistilBERT/attack_log.json    # Logs padrão
tail -f logs/TinyBERT/tinybert_attack_log.json  # Logs IoT otimizados  
tail -f logs/MiniLM/minilm_attack_log.json      # Logs workstation

# Verificar erros específicos
grep -i error logs/*/*.json
```

## 🌐 Integração e APIs

### Uso Programático Unificado

```python
# Todos os modelos agora têm interface consistente
import sys

# Para DistilBERT
sys.path.append('DistilBERT')
from realtime_network_monitor import NetworkAttackDetector
detector_distilbert = NetworkAttackDetector('model.onnx', 'metadata.pkl')

# Para TinyBERT (otimizado IoT)
sys.path.append('TinyBERT')  
from realtime_network_monitor import NetworkAttackDetector
detector_tinybert = NetworkAttackDetector('model.onnx', 'metadata.pkl')

# Para MiniLM (balanceado)
sys.path.append('MiniLM')
from realtime_network_monitor import NetworkAttackDetector
detector_minilm = NetworkAttackDetector('model.onnx', 'metadata.pkl')

# Interface idêntica para todos:
result = detector.predict(network_features)
print(f"Modelo: {result['model']}")
print(f"É ataque: {result['is_attack']}")
print(f"Confiança: {result['confidence']}")
```

### Saída Padronizada

Todos os modelos retornam o mesmo formato:

```json
{
    "timestamp": "2024-01-15T10:30:00",
    "model": "TinyBERT|MiniLM|DistilBERT",
    "predicted_class": "DDoS-SlowHTTPTest",
    "confidence": 0.95,
    "is_attack": true,
    "is_benign": false,
    "is_high_threat": true,
    "is_ddos": true,
    "inference_time_ms": 2.5,
    "memory_usage_mb": 150.2,
    "cpu_usage_percent": 45.1
}
```

**Para tráfego benigno:**
```json
{
    "predicted_class": "BenignTraffic",
    "confidence": 0.95,
    "is_attack": false,
    "is_benign": true,
    "is_high_threat": false,
    "is_ddos": false
}
```

## 📈 Performance e Benchmarks

### Executar Benchmarks Comparativos

```bash
# Comparar todos os modelos
./benchmark_all_models.sh

# Benchmark individual com detalhes
cd TinyBERT
python3 realtime_network_monitor.py --benchmark  # IoT metrics
cd ../MiniLM  
python3 realtime_network_monitor.py --benchmark  # Workstation metrics
cd ../DistilBERT
python3 realtime_network_monitor.py --benchmark  # Server metrics
```

### Métricas Coletadas por Modelo

#### TinyBERT (IoT)
- Latência ultra-baixa (< 5ms alvo)
- Uso mínimo de memória (< 200MB)
- Cache otimizado (50 entradas)
- Garbage collection agressiva

#### MiniLM (Workstation)  
- Latência balanceada (< 15ms)
- Uso moderado de memória (< 500MB)
- Métricas P95/P99 de latência
- Monitoramento de CPU por core

#### DistilBERT (Servidor)
- Precisão máxima
- Análise detalhada de ameaças
- Performance consistente
- Logging extensivo

## 🔒 Segurança

### Boas Práticas

- Execute com usuário não-root sempre que possível
- Monitore logs regularmente
- Mantenha o sistema atualizado
- Configure alertas para detecções críticas
- Faça backup das configurações
- **Use o modelo apropriado para seu ambiente**:
  - **IoT/Edge**: TinyBERT
  - **Workstation/Desktop**: MiniLM  
  - **Servidor/Análise**: DistilBERT

### Rotação de Logs

```bash
# Rotação automática (adicionar ao crontab)
0 0 * * * find logs/ -name "*.json" -mtime +30 -delete

# Rotação manual por modelo
find logs/TinyBERT/ -name "*.json" -mtime +7 -delete    # IoT: 7 dias
find logs/MiniLM/ -name "*.json" -mtime +15 -delete     # Workstation: 15 dias  
find logs/DistilBERT/ -name "*.json" -mtime +30 -delete # Servidor: 30 dias
```

## 📞 Suporte

### Informações do Sistema

```bash
# Informações após instalação
./setup_environment.sh  # mostra resumo final

# Verificação manual de cada modelo
source venv/bin/activate

cd DistilBERT && python3 -c "from realtime_network_monitor import NetworkAttackDetector; print('DistilBERT: OK')" && cd ..
cd TinyBERT && python3 -c "from realtime_network_monitor import NetworkAttackDetector; print('TinyBERT: OK')" && cd ..  
cd MiniLM && python3 -c "from realtime_network_monitor import NetworkAttackDetector; print('MiniLM: OK')" && cd ..
```

### Debug por Modelo

```bash
# Debug TinyBERT (IoT)
cd TinyBERT
python3 realtime_network_monitor.py --benchmark  # Verificar latência
python3 -c "import psutil; print(f'RAM: {psutil.virtual_memory().available/1024/1024:.0f}MB')"

# Debug MiniLM (Workstation)
cd MiniLM  
python3 realtime_network_monitor.py --benchmark  # Verificar performance balanceada
python3 -c "import psutil; print(f'CPUs: {psutil.cpu_count()}')"

# Debug DistilBERT (Servidor)
cd DistilBERT
python3 realtime_network_monitor.py --benchmark  # Verificar precisão
```

### Contribuição

Para melhorias ou problemas:
1. Verifique os logs de erro por modelo
2. Execute benchmark do modelo específico
3. Documente ambiente (IoT/Workstation/Servidor)
4. Inclua informações do sistema (OS, Python, RAM, CPU)

---

**Sistema unificado para detecção eficiente de ataques de rede usando modelos BERT otimizados** 🛡️

**Modelos otimizados para diferentes ambientes: IoT (TinyBERT), Workstation (MiniLM), Servidor (DistilBERT)** ⚡ 