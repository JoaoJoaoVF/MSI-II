# Guia de Execução dos Monitores - Versão Otimizada

## ⚠️ Problema Identificado e Solução

O problema que você enfrentou foi **falta de espaço em disco** (`OSError: [Errno 28] No space left on device`). Isso explica por que muitos arquivos de resultado ficaram vazios - o script conseguiu criar os arquivos mas não conseguiu escrever neles.

## 📋 Scripts Atualizados

### 1. `run_all_monitors.sh` (Linux/Mac)

Script Bash otimizado com:

- ✅ Verificação de espaço em disco antes de iniciar
- ✅ Monitoramento contínuo de espaço durante execução
- ✅ Limpeza automática de arquivos temporários
- ✅ Processamento em lotes menores
- ✅ Timeout de 5 minutos por arquivo
- ✅ Relatórios de progresso detalhados
- ✅ Resumos consolidados em `analysis_results/`

### 2. `run_all_monitors.ps1` (Windows)

Script PowerShell equivalente com as mesmas funcionalidades.

### 3. `manage_results.sh` (Linux/Mac)

Script para gerenciamento avançado de resultados:

- 🧹 Limpeza de arquivos vazios
- 📦 Compactação de arquivos grandes
- 💾 Backup automático
- 📚 Arquivamento de resultados antigos
- 📊 Estatísticas detalhadas

## 🚀 Como Usar

### Opção 1: Modo Interativo (Recomendado)

#### Linux/Mac:

```bash
./run_all_monitors.sh
```

#### Windows:

```powershell
.\run_all_monitors.ps1
```

### Opção 2: Linha de Comando

#### Linux/Mac:

```bash
# Executar todos os modelos
./run_all_monitors.sh --all

# Executar modelo específico
./run_all_monitors.sh --distilbert
./run_all_monitors.sh --minilm
./run_all_monitors.sh --tinybert

# Limpeza
./run_all_monitors.sh --cleanup

# Verificar espaço
./run_all_monitors.sh --disk
```

#### Windows:

```powershell
# Executar todos os modelos
.\run_all_monitors.ps1 -Action all

# Executar modelo específico
.\run_all_monitors.ps1 -Action distilbert
.\run_all_monitors.ps1 -Action minilm
.\run_all_monitors.ps1 -Action tinybert

# Limpeza
.\run_all_monitors.ps1 -Action cleanup

# Verificar espaço
.\run_all_monitors.ps1 -Action disk
```

## 📊 Estrutura de Resultados

```
analysis_results/
├── DistilBERT/
│   └── processing_summary.txt
├── MiniLM/
│   └── processing_summary.txt
├── TinyBERT/
│   └── processing_summary.txt
└── consolidated_summary.txt
```

## 🔧 Melhorias Implementadas

### 1. Controle de Espaço em Disco

- Verificação inicial: mínimo 5GB livre
- Monitoramento contínuo: limpeza automática quando < 500MB
- Interrupção de emergência quando < 200MB

### 2. Processamento Otimizado

- Lotes de 10 arquivos por vez
- Timeout de 5 minutos por arquivo
- Retry automático em caso de erro de espaço

### 3. Limpeza Automática

- Remove arquivos vazios (0 bytes)
- Remove arquivos muito grandes (>100MB)
- Compacta logs de ataque grandes (>5MB)

### 4. Relatórios Detalhados

- Estatísticas de sucesso/erro por modelo
- Tempo de processamento
- Tamanho dos arquivos gerados
- Resumo consolidado

## 🛠️ Antes de Executar

### 1. Verificar Espaço Disponível

```bash
# Linux/Mac
df -h .

# Windows
Get-WmiObject -Class Win32_LogicalDisk | Select-Object DeviceID, @{Name="Size(GB)";Expression={[math]::Round($_.Size/1GB,2)}}, @{Name="FreeSpace(GB)";Expression={[math]::Round($_.FreeSpace/1GB,2)}}
```

### 2. Limpar Arquivos Antigos (Se Necessário)

```bash
# Linux/Mac
./manage_results.sh --full-clean

# Windows - executar manualmente:
# Remove arquivos vazios
Get-ChildItem -Recurse -Name "result-*.txt" | Where-Object { (Get-Item $_).Length -eq 0 } | Remove-Item

# Remove arquivos muito grandes
Get-ChildItem -Recurse -Name "result-*.txt" | Where-Object { (Get-Item $_).Length -gt 100MB } | Remove-Item
```

### 3. Fazer Backup (Opcional)

```bash
# Linux/Mac
tar -czf backup_results_$(date +%Y%m%d).tar.gz */result-*.txt */attack_log.json

# Windows
Compress-Archive -Path "*/result-*.txt", "*/attack_log.json" -DestinationPath "backup_results_$(Get-Date -Format 'yyyyMMdd').zip"
```

## 📈 Monitoramento Durante Execução

O script agora mostra:

- Progresso em tempo real: `[15/106] Processando: arquivo.csv`
- Status de espaço: `💾 Espaço disponível: 12.5GB`
- Limpeza automática: `🧹 Limpando arquivos temporários...`
- Resultados: `✅ Concluído: arquivo.csv (1.2MB)`

## 🚨 Solução para Problemas Comuns

### 1. Erro de Espaço em Disco

```bash
# Verificar espaço
df -h .

# Limpeza manual
find . -name "result-*.txt" -size 0 -delete
find . -name "result-*.txt" -size +100M -delete

# Mover arquivos para outro local
mkdir /outro/local/backup
mv */result-*.txt /outro/local/backup/
```

### 2. Muitos Arquivos Vazios

```bash
# Usar o script de limpeza
./manage_results.sh --clean

# Ou manualmente
find . -name "result-*.txt" -size 0 -delete
```

### 3. Processo Muito Lento

- Use processamento sequencial em vez de paralelo
- Reduza o `BATCH_SIZE` de 10 para 5
- Monitore CPU e memória: `htop` (Linux) ou Task Manager (Windows)

## 📝 Logs e Depuração

### Localização dos Logs

- Resumos: `analysis_results/[MODELO]/processing_summary.txt`
- Logs de ataque: `[MODELO]/attack_log.json`
- Resultados individuais: `[MODELO]/result-*.txt`

### Verificar Status

```bash
# Ver resumo de um modelo
cat analysis_results/DistilBERT/processing_summary.txt

# Contar sucessos/erros
grep -c "SUCESSO" analysis_results/*/processing_summary.txt
grep -c "ERRO" analysis_results/*/processing_summary.txt

# Ver arquivos maiores
find . -name "result-*.txt" -size +1M -exec ls -lh {} \;
```

## 🎯 Próximos Passos Recomendados

1. **Execute primeiro**: `./run_all_monitors.sh --disk` para verificar espaço
2. **Se pouco espaço**: `./run_all_monitors.sh --cleanup` para limpar
3. **Execute teste**: Comece com um modelo só para testar
4. **Monitore**: Acompanhe os logs em tempo real
5. **Análise**: Use os resumos em `analysis_results/` para análise

## 💡 Dicas Importantes

- ⚡ **Execução sequencial é mais segura** que paralela para grandes volumes
- 🧹 **Limpeza regular** evita problemas de espaço
- 📊 **Monitore o progresso** através dos arquivos de resumo
- 💾 **Mantenha pelo menos 10GB livres** para execução paralela
- 🔄 **Faça backups** dos resultados importantes antes de limpezas

Com essas melhorias, você deve conseguir processar todos os arquivos sem problemas de espaço em disco!

## 🎯 VERSÃO COMPLETA - Garantia de Métricas Completas

### ⭐ Nova Funcionalidade Principal: Coleta Completa de Métricas

O script foi **completamente reconfigurado** para garantir que:

✅ **TODOS os 3 modelos** (DistilBERT, MiniLM, TinyBERT) processem **TODOS os arquivos CSV**  
✅ **TODAS as métricas e scores** sejam coletadas de cada arquivo  
✅ **Casos de predição incorreta** sejam registrados em detalhes  
✅ **Nenhum arquivo seja pulado** mesmo com problemas de espaço  
✅ **Relatórios detalhados** sejam gerados com estatísticas completas

### 📊 Arquivos de Métricas Gerados

Para cada modelo, são criados:

1. **`processing_summary.txt`** - Resumo geral do processamento
2. **`metrics_summary.txt`** - Métricas detalhadas por arquivo CSV:

   - Status do processamento
   - Tempo de execução
   - Predições corretas/incorretas
   - Accuracy, Precision, Recall, F1-Score
   - Tamanho do arquivo de resultado

3. **`failed_predictions.txt`** - Casos onde o modelo errou:

   - Detalhes das predições incorretas
   - Tipos de ataques mal classificados
   - Estatísticas de falhas por categoria

4. **`consolidated_statistics.txt`** - Estatísticas consolidadas
5. **`complete_execution_summary.txt`** - Resumo da execução completa

### 🔍 Script de Análise de Métricas

Novo script `analyze_metrics.sh` para análise detalhada:

```bash
# Analisar todos os modelos
./analyze_metrics.sh --all

# Comparar performance entre modelos
./analyze_metrics.sh --compare

# Ver arquivos faltantes
./analyze_metrics.sh --missing

# Resumo rápido
./analyze_metrics.sh --summary
```

### 📋 Estrutura Completa de Resultados

```
analysis_results/
├── DistilBERT/
│   ├── processing_summary.txt     # Resumo do processamento
│   ├── metrics_summary.txt        # Métricas detalhadas (CSV format)
│   └── failed_predictions.txt     # Predições incorretas
├── MiniLM/
│   ├── processing_summary.txt
│   ├── metrics_summary.txt
│   └── failed_predictions.txt
├── TinyBERT/
│   ├── processing_summary.txt
│   ├── metrics_summary.txt
│   └── failed_predictions.txt
├── complete_execution_summary.txt  # Resumo geral
└── consolidated_statistics.txt     # Estatísticas consolidadas

analysis_reports/
├── DistilBERT_analysis_report.txt  # Análise detalhada
├── MiniLM_analysis_report.txt
├── TinyBERT_analysis_report.txt
├── models_comparison.txt           # Comparação entre modelos
└── missing_files_report.txt        # Arquivos faltantes
```
