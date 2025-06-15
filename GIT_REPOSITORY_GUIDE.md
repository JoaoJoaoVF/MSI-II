# 📦 Guia para Repositório Git - MSI-II

## 🎯 **RESUMO EXECUTIVO**

Para um repositório Git limpo e funcional, você precisa manter **apenas 37 arquivos essenciais** de um total de mais de 100 arquivos no projeto.

---

## ✅ **ARQUIVOS OBRIGATÓRIOS PARA O GIT**

### **📁 Raiz do Projeto (16 arquivos):**
```
✅ MANTER NO GIT:
├── .gitignore                           # Configuração Git
├── .gitattributes                       # Configuração Git
├── README.md                            # Documentação principal
├── README_SETUP.md                      # Guia de instalação
├── README_EXECUTION.md                  # Guia de execução
├── requirements_consolidated.txt        # Dependências consolidadas
├── setup_environment.sh                # Script de configuração principal
├── run_all_monitors.sh                 # Script de execução principal
├── start_all_models.sh                 # Gerenciamento unificado
├── stop_all_models.sh                  # Gerenciamento unificado
├── benchmark_all_models.sh             # Benchmark unificado
├── manage_results.sh                   # Gerenciamento de resultados
├── analyze_metrics.sh                  # Análise de métricas
├── cleanup_models.sh                   # Script de limpeza
├── CLEANUP_GUIDE.md                    # Guia de limpeza
└── part-00002-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv  # 1 arquivo CSV exemplo
```

### **📁 DistilBERT/ (5 arquivos):**
```
✅ MANTER NO GIT:
├── realtime_network_monitor.py          # Script principal
├── performance_analyzer.py              # Análise de performance
├── network_attack_detector_quantized.onnx  # Modelo ONNX
├── model_metadata.pkl                   # Metadados
└── requirements.txt                     # Dependências específicas
```

### **📁 MiniLM/ (5 arquivos):**
```
✅ MANTER NO GIT:
├── realtime_network_monitor.py          # Script principal
├── minilm_network_monitor.py           # Monitor específico
├── minilm_attack_detector_quantized.onnx  # Modelo ONNX
├── minilm_metadata.pkl                 # Metadados
└── requirements.txt                     # Dependências específicas
```

### **📁 TinyBERT/ (5 arquivos):**
```
✅ MANTER NO GIT:
├── realtime_network_monitor.py          # Script principal
├── tinybert_network_monitor.py         # Monitor específico
├── tinybert_attack_detector_quantized.onnx  # Modelo ONNX
├── tinybert_metadata.pkl               # Metadados
└── requirements.txt                     # Dependências específicas
```

### **📁 config/ (3 arquivos):**
```
✅ MANTER NO GIT:
├── distilbert_config.json              # Configuração DistilBERT
├── minilm_config.json                  # Configuração MiniLM
└── tinybert_config.json                # Configuração TinyBERT
```

### **📁 data/ (1 arquivo exemplo):**
```
✅ MANTER NO GIT:
└── part-00002-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv  # Arquivo exemplo
```

---

## 🗑️ **ARQUIVOS PARA REMOVER DO GIT**

### **📁 Raiz - Scripts redundantes:**
```
🗑️ REMOVER DO GIT:
├── start_distilbert.sh                 # Redundante (gerado pelo setup_environment.sh)
├── start_minilm.sh                     # Redundante (gerado pelo setup_environment.sh)
├── start_tinybert.sh                   # Redundante (gerado pelo setup_environment.sh)
├── simulate_distilbert.sh              # Redundante (gerado pelo setup_environment.sh)
├── simulate_minilm.sh                  # Redundante (gerado pelo setup_environment.sh)
├── simulate_tinybert.sh                # Redundante (gerado pelo setup_environment.sh)
├── benchmark_distilbert.sh             # Redundante (gerado pelo setup_environment.sh)
├── benchmark_minilm.sh                 # Redundante (gerado pelo setup_environment.sh)
└── benchmark_tinybert.sh               # Redundante (gerado pelo setup_environment.sh)
```

### **📁 data/ - Dados desnecessários:**
```
🗑️ REMOVER DO GIT:
├── extract_attack_types.py             # Script de processamento de dados
├── link_data_files.txt                 # Lista de arquivos
└── part-*.csv (todos exceto 1 exemplo) # Dados grandes (manter apenas 1 exemplo)
```

### **📁 Pastas dos modelos - Arquivos temporários:**
```
🗑️ REMOVER DO GIT (de cada pasta DistilBERT/, MiniLM/, TinyBERT/):
├── part-*.csv                          # Dados duplicados
├── result-*.txt                        # Resultados de execuções
├── attack_log.json                     # Logs antigos
├── *_attack_log.json                   # Logs específicos
├── *_optimization.ipynb                # Notebooks de desenvolvimento
├── test_*.py                           # Scripts de teste
├── run_*.sh                            # Scripts redundantes
├── start_detector.sh                   # Scripts específicos
├── config_environment.sh               # Configurações específicas
├── network-detector.service            # Configuração de serviço
├── device_configs.sh                   # Configurações de dispositivo
├── install_*.sh                        # Scripts de instalação
├── requirements-*.txt                  # Requirements específicos (exceto requirements.txt)
└── config/                             # Subdiretórios de configuração
```

### **📁 Diretórios a remover/ignorar:**
```
🗑️ REMOVER/IGNORAR NO GIT:
├── venv/                               # Ambiente virtual Python
├── logs/                               # Logs de execução
├── analysis_results/                   # Resultados de análises
├── backup_*/                           # Backups automáticos
├── archived_results/                   # Resultados arquivados
├── __pycache__/                        # Cache Python
└── .git/ (manter apenas se for o repo principal)
```

---

## 📝 **.gitignore Recomendado**

Crie/atualize o arquivo `.gitignore` com:

```gitignore
# Ambiente Python
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
pip-log.txt
pip-delete-this-directory.txt

# Resultados e logs
logs/
analysis_results/
analysis_reports/
archived_results/
backup_*/
*.log
*.tmp
attack_log*.json

# Arquivos de resultado
result-*.txt
*.pid

# Dados grandes (manter apenas exemplos)
data/part-*.csv
!data/part-00002-*.csv

# Arquivos temporários
*.bak
*.tmp
*~

# Scripts gerados automaticamente
start_distilbert.sh
start_minilm.sh
start_tinybert.sh
simulate_*.sh
benchmark_distilbert.sh
benchmark_minilm.sh
benchmark_tinybert.sh

# Configurações locais
config_environment.sh
device_configs.sh
install_*.sh

# Jupyter Notebooks de desenvolvimento
*_optimization.ipynb

# Arquivos do sistema
.DS_Store
Thumbs.db
```

---

## 🚀 **COMANDOS PARA LIMPEZA**

### **1. Remover arquivos desnecessários:**
```bash
# Remover scripts redundantes
rm -f start_distilbert.sh start_minilm.sh start_tinybert.sh
rm -f simulate_*.sh benchmark_distilbert.sh benchmark_minilm.sh benchmark_tinybert.sh

# Limpar dados grandes (manter apenas 1 exemplo)
cd data/
find . -name "part-*.csv" ! -name "part-00002-*" -delete
rm -f extract_attack_types.py link_data_files.txt
cd ..

# Limpar pastas dos modelos
for model in DistilBERT MiniLM TinyBERT; do
    cd $model/
    rm -f part-*.csv result-*.txt *attack_log.json
    rm -f *_optimization.ipynb test_*.py run_*.sh start_detector.sh
    rm -f config_environment.sh device_configs.sh install_*.sh
    rm -f requirements-*.txt requirements_*.txt
    rm -f network-detector.service
    rm -rf config/
    cd ..
done

# Remover diretórios temporários
rm -rf venv/ logs/ analysis_results/ backup_* archived_results/
```

### **2. Atualizar Git:**
```bash
# Adicionar .gitignore atualizado
git add .gitignore

# Remover arquivos já trackeados que agora devem ser ignorados
git rm --cached -r logs/ analysis_results/ venv/ 2>/dev/null || true
git rm --cached start_distilbert.sh start_minilm.sh start_tinybert.sh 2>/dev/null || true
git rm --cached simulate_*.sh benchmark_distilbert.sh benchmark_minilm.sh benchmark_tinybert.sh 2>/dev/null || true

# Commit das mudanças
git add .
git commit -m "🧹 Limpeza do repositório: mantidos apenas arquivos essenciais"
```

---

## 📊 **ESTATÍSTICAS**

### **Antes da limpeza:**
- **Total**: ~100+ arquivos
- **Tamanho**: ~500MB+ (com dados e resultados)
- **Scripts**: 20+ scripts redundantes

### **Depois da limpeza:**
- **Total**: 37 arquivos essenciais
- **Tamanho**: ~50-100MB (apenas essenciais)
- **Scripts**: 8 scripts principais

### **Benefícios:**
- ✅ **90% menos arquivos** no repositório
- ✅ **80% menos espaço** utilizado
- ✅ **Clone mais rápido** do repositório
- ✅ **Commits mais limpos** e focados
- ✅ **Documentação organizada**
- ✅ **Fácil de manter** e versionar

---

## 🔧 **IMPORTANTE PARA DESENVOLVEDORES**

### **O que acontece quando alguém clona o repo:**
1. Clone contém apenas arquivos essenciais (37 arquivos)
2. Executa `./setup_environment.sh` que:
   - Cria ambiente virtual
   - Instala dependências
   - Gera scripts auxiliares automaticamente
   - Cria diretórios necessários
3. Sistema fica 100% funcional

### **Scripts que são recriados automaticamente:**
- `start_*.sh` - Gerados pelo `setup_environment.sh`
- `simulate_*.sh` - Gerados pelo `setup_environment.sh`
- `benchmark_*.sh` - Gerados pelo `setup_environment.sh`
- Diretórios `logs/`, `analysis_results/`, etc.

### **Arquivo exemplo de dados:**
- Mantido 1 arquivo CSV como exemplo
- Usuário adiciona seus próprios dados na pasta `data/`
- Sistema detecta automaticamente novos arquivos

---

## 💡 **DICAS PARA MANUTENÇÃO**

1. **Sempre use `.gitignore`** para evitar commit de arquivos temporários
2. **Mantenha apenas 1-2 arquivos CSV** como exemplo
3. **Scripts auxiliares** são gerados automaticamente
4. **Logs e resultados** nunca devem ir para o Git
5. **Ambiente virtual** sempre deve ser ignorado
6. **Backups automáticos** devem ser locais apenas

---

## 🎯 **RESULTADO FINAL**

Seu repositório Git ficará com **37 arquivos essenciais** que permitem:
- ✅ Qualquer pessoa clonar e usar o sistema
- ✅ Instalação automática com `setup_environment.sh`
- ✅ Execução completa com `run_all_monitors.sh`
- ✅ Todos os 3 modelos funcionando
- ✅ Documentação completa
- ✅ Exemplos de uso
- ✅ Repositório limpo e profissional
