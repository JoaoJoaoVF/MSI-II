# Sistema de Detecção de Ataques de Rede com MiniLM

Sistema baseado em MiniLM (Microsoft's Multilingual Language Model) otimizado para detectar ataques DDoS em tempo real em dispositivos IoT como Raspberry Pi.

## Características do Modelo MiniLM

- **Modelo**: MiniLM adaptado para dados tabulares
- **Parâmetros**: ~22M (vs 66M do DistilBERT)
- **Otimizações**: Quantização INT8, ONNX Runtime
- **Tamanho**: ~15-20 MB após quantização
- **Performance**: ~5-8 ms por predição
- **Vantagens**: Menor uso de memória, inferência mais rápida

## Vantagens do MiniLM sobre DistilBERT

1. **Menor Tamanho**: 67% menor que DistilBERT
2. **Velocidade**: 2-3x mais rápido na inferência
3. **Memória**: Usa 60% menos RAM
4. **Eficiência**: Ideal para dispositivos com recursos limitados
5. **Multilingual**: Suporte nativo a múltiplos idiomas

## Arquitetura Otimizada

- **Layers**: 2 (vs 6 do DistilBERT)
- **Hidden Size**: 128 (vs 768 do BERT)
- **Attention Heads**: 4 (vs 12 do BERT)
- **Vocabulary**: 1000 tokens (ultra-compacto)

## Instalação no Raspberry Pi

```bash
# Clonar repositório
git clone <repo-url>
cd MiniLM

# Instalar dependências
pip install -r requirements.txt

# Executar benchmark
python3 minilm_network_monitor.py --benchmark
```

## Uso

### Treinamento
```bash
jupyter notebook MiniLM_optimization.ipynb
```

### Monitoramento em Tempo Real
```bash
python3 minilm_network_monitor.py --simulate dados.csv --delay 0.1
```

### Benchmark de Performance
```bash
python3 minilm_network_monitor.py --benchmark
```

## Comparação de Performance

| Modelo | Parâmetros | Tamanho | Inferência | Memória |
|--------|------------|---------|------------|---------|
| BERT | 110M | ~440MB | ~50ms | ~2GB |
| DistilBERT | 66M | ~260MB | ~25ms | ~1GB |
| **MiniLM** | **22M** | **~90MB** | **~8ms** | **~300MB** |
| TinyBERT | 14.5M | ~60MB | ~5ms | ~200MB |

## Tipos de Ataques Detectados

- DDoS (Distributed Denial of Service)
- DoS (Denial of Service)
- Port Scan
- Brute Force
- Web Attack
- Infiltration
- Botnet

## Arquivos Incluídos

- `MiniLM_optimization.ipynb`: Notebook principal
- `minilm_network_monitor.py`: Sistema de monitoramento
- `requirements.txt`: Dependências Python
- `README.md`: Esta documentação

## Métricas Esperadas

- **Accuracy**: >95%
- **Precision**: >94%
- **Recall**: >93%
- **F1-Score**: >94%
- **Latência**: <10ms
- **Throughput**: >100 predições/segundo

## Integração IoT

O MiniLM é especialmente adequado para:
- Raspberry Pi 3B+ ou superior
- NVIDIA Jetson Nano
- Edge devices com ARM
- Sistemas embarcados
- Gateways IoT

## Suporte

Para dúvidas ou problemas:
1. Verifique os logs do sistema
2. Confirme compatibilidade das features
3. Teste com dados de exemplo
4. Consulte a documentação do transformers 