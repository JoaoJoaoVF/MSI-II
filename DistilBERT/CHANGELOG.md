# Changelog - Sistema de Detecção de Ataques

## [1.2.0] - 2024-12-19

### 🐛 Correções de Bugs
- **CRÍTICO**: Corrigido erro de sintaxe `SyntaxError: unterminated string literal` no `realtime_network_monitor.py`
  - Linha 116: Corrigida string literal não terminada em `log_detection()`
  - Linha 179+: Corrigidas múltiplas f-strings com quebras de linha incorretas
- **CRÍTICO**: Corrigido erro de dependência `ERROR: No matching distribution found for onnxruntime==1.15.1`
  - Atualizado `requirements.txt` para usar versões flexíveis (`>=1.17.0`)
  - Criado `requirements-raspberry.txt` específico para Raspberry Pi

### ✨ Novas Funcionalidades
- **Novo**: Script `test_syntax.py` para verificação automática de sintaxe
- **Novo**: Script `install_manual.sh` para instalação manual de dependências
- **Novo**: Guia completo `TROUBLESHOOTING.md` com soluções para problemas comuns
- **Melhorado**: Script `deploy_raspberry_pi.sh` com instalação robusta e fallbacks
- **NOVO**: Geração automática de arquivos de resultado baseados no nome do CSV
  - Formato: `result-nome_do_csv.txt`
  - Inclui estatísticas completas, tipos de ataques detectados e detalhes de cada amostra
  - Opção `--output` para especificar arquivo de saída personalizado
- **MELHORADO**: Relatórios detalhados com resumo executivo e análise por tipo de ataque

### 🔧 Melhorias
- **Instalação**: Sistema de instalação mais robusto com múltiplas opções de fallback
- **Documentação**: README atualizado com instruções claras de solução de problemas
- **Testes**: Script de teste de instalação mais abrangente
- **Compatibilidade**: Suporte para versões mais recentes do ONNX Runtime
- **Interface**: Resultados não são mais apenas exibidos na tela, mas salvos em arquivos estruturados
- **Análise**: Relatórios incluem estatísticas por tipo de ataque e timeline das detecções
- **Usabilidade**: Arquivo de resultado é criado automaticamente sem configuração adicional

### 📋 Arquivos Modificados
- `realtime_network_monitor.py` - Correções de sintaxe
- `requirements.txt` - Versões atualizadas
- `requirements-raspberry.txt` - Novo arquivo para Raspberry Pi
- `deploy_raspberry_pi.sh` - Instalação robusta
- `README.md` - Documentação atualizada
- `TROUBLESHOOTING.md` - Novo guia de problemas

### 📋 Arquivos Novos
- `install_manual.sh` - Script de instalação manual
- `test_syntax.py` - Verificador de sintaxe
- `test_installation.py` - Teste completo de instalação
- `CHANGELOG.md` - Este arquivo

### 🎯 Como Atualizar
```bash
# 1. Baixar arquivos atualizados
# 2. Verificar sintaxe
python test_syntax.py

# 3. Testar instalação
python test_installation.py

# 4. Se houver problemas, usar instalação manual
./install_manual.sh
```

### 🚀 Próximas Versões
- [ ] Suporte para modelos ONNX mais recentes
- [ ] Interface web para monitoramento
- [ ] Integração com sistemas de alertas
- [ ] Otimizações para Raspberry Pi Zero
- [ ] Suporte para múltiplos modelos simultâneos 

## [1.1.0] - 2024-12-19

### 🐛 Correções de Bugs
- **CRÍTICO**: Corrigido erro de sintaxe `SyntaxError: unterminated string literal` no `realtime_network_monitor.py`
  - Linha 116: Corrigida string literal não terminada em `log_detection()`
  - Linha 179+: Corrigidas múltiplas f-strings com quebras de linha incorretas
- **CRÍTICO**: Corrigido erro de dependência `ERROR: No matching distribution found for onnxruntime==1.15.1`
  - Atualizado `requirements.txt` para usar versões flexíveis (`>=1.17.0`)
  - Criado `requirements-raspberry.txt` específico para Raspberry Pi

### ✨ Novas Funcionalidades
- **Novo**: Script `test_syntax.py` para verificação automática de sintaxe
- **Novo**: Script `install_manual.sh` para instalação manual de dependências
- **Novo**: Guia completo `TROUBLESHOOTING.md` com soluções para problemas comuns
- **Melhorado**: Script `deploy_raspberry_pi.sh` com instalação robusta e fallbacks

### 🔧 Melhorias
- **Instalação**: Sistema de instalação mais robusto com múltiplas opções de fallback
- **Documentação**: README atualizado com instruções claras de solução de problemas
- **Testes**: Script de teste de instalação mais abrangente
- **Compatibilidade**: Suporte para versões mais recentes do ONNX Runtime

### 📋 Arquivos Modificados
- `realtime_network_monitor.py` - Correções de sintaxe
- `requirements.txt` - Versões atualizadas
- `requirements-raspberry.txt` - Novo arquivo para Raspberry Pi
- `deploy_raspberry_pi.sh` - Instalação robusta
- `README.md` - Documentação atualizada
- `TROUBLESHOOTING.md` - Novo guia de problemas

### 📋 Arquivos Novos
- `install_manual.sh` - Script de instalação manual
- `test_syntax.py` - Verificador de sintaxe
- `test_installation.py` - Teste completo de instalação
- `CHANGELOG.md` - Este arquivo

### 🎯 Como Atualizar
```bash
# 1. Baixar arquivos atualizados
# 2. Verificar sintaxe
python test_syntax.py

# 3. Testar instalação
python test_installation.py

# 4. Se houver problemas, usar instalação manual
./install_manual.sh
```

### 🚀 Próximas Versões
- [ ] Suporte para modelos ONNX mais recentes
- [ ] Interface web para monitoramento
- [ ] Integração com sistemas de alertas
- [ ] Otimizações para Raspberry Pi Zero
- [ ] Suporte para múltiplos modelos simultâneos 