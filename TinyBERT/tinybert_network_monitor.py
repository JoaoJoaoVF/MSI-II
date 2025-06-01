#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Detecção de Ataques de Rede em Tempo Real - TinyBERT
Ultra-otimizado para dispositivos IoT e microcontroladores
"""

import onnxruntime as ort
import numpy as np
import pandas as pd
import pickle
import time
import argparse
import json
from datetime import datetime
import threading
import queue
import sys
import psutil
import os
import gc

class TinyBERTNetworkDetector:
    def __init__(self, model_path, metadata_path):
        """Inicializar detector de ataques TinyBERT"""
        
        print("🚀 Carregando modelo TinyBERT ultra-otimizado...")
        
        # Configurar ONNX Runtime para máxima eficiência
        providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = min(psutil.cpu_count(), 2)  # Limitar threads
        sess_options.inter_op_num_threads = 1  # Single thread para IoT
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        self.session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=providers
        )
        
        print("📊 Carregando metadados...")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.scaler = self.metadata['scaler']
        self.label_encoder = self.metadata['label_encoder']
        self.feature_names = self.metadata['feature_names']
        self.classes = self.metadata['classes']
        
        print(f"✅ Modelo TinyBERT carregado com sucesso!")
        print(f"📋 Classes detectáveis: {self.classes}")
        print(f"🔧 Features: {len(self.feature_names)}")
        print(f"💾 Uso de memória inicial: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")
        
        # Estatísticas de performance ultra-detalhadas
        self.total_predictions = 0
        self.attack_detections = 0
        self.inference_times = []
        self.memory_usage = []
        self.cpu_usage = []
        
        # Cache para otimização
        self._feature_cache = {}
        self._prediction_cache = {}
        
        # Warm-up do modelo
        self._warmup()
    
    def _warmup(self):
        """Warm-up ultra-rápido do modelo"""
        print("🔥 Aquecendo TinyBERT...")
        dummy_features = {name: 0.0 for name in self.feature_names}  # Usar zeros para consistência
        
        # Warm-up mínimo para TinyBERT
        for _ in range(5):
            self.predict(dummy_features, verbose=False)
        
        # Limpeza de memória após warm-up
        gc.collect()
        print("✅ Warm-up concluído!")
    
    def preprocess_features(self, features_dict):
        """Pré-processar features com cache otimizado"""
        
        # Criar chave de cache
        cache_key = hash(tuple(sorted(features_dict.items())))
        
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        # Converter para array na ordem correta
        features_array = np.array([features_dict.get(name, 0.0) for name in self.feature_names])
        
        # Normalizar usando o scaler treinado
        features_scaled = self.scaler.transform(features_array.reshape(1, -1))
        features_scaled = features_scaled.astype(np.float32)
        
        # Cache limitado para evitar uso excessivo de memória
        if len(self._feature_cache) < 100:
            self._feature_cache[cache_key] = features_scaled
        
        return features_scaled
    
    def predict(self, features_dict, verbose=True):
        """Fazer predição ultra-rápida"""
        
        # Medir recursos antes da inferência
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_before = process.cpu_percent()
        
        # Pré-processar
        features = self.preprocess_features(features_dict)
        
        # Inferência ultra-otimizada
        start_time = time.perf_counter()  # Maior precisão
        ort_inputs = {'features': features}
        logits, probabilities = self.session.run(None, ort_inputs)
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Medir recursos após a inferência
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        cpu_after = process.cpu_percent()
        
        # Processar resultado
        predicted_class_idx = np.argmax(probabilities[0])
        predicted_class = self.classes[predicted_class_idx]
        confidence = probabilities[0][predicted_class_idx]
        
        # Atualizar estatísticas
        self.total_predictions += 1
        self.inference_times.append(inference_time)
        self.memory_usage.append(memory_after)
        self.cpu_usage.append(cpu_after)
        
        is_attack = predicted_class.lower() != 'benign'
        if is_attack:
            self.attack_detections += 1
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'model': 'TinyBERT',
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'is_attack': is_attack,
            'inference_time_ms': inference_time,
            'memory_usage_mb': memory_after,
            'cpu_usage_percent': cpu_after,
            'all_probabilities': probabilities[0].tolist()
        }
        
        # Alertas para dispositivos IoT
        if verbose:
            if inference_time > 5:  # Alerta para TinyBERT se > 5ms
                print(f"⚠️ Latência alta para TinyBERT: {inference_time:.2f}ms")
            if memory_after > 300:  # Alerta se > 300MB
                print(f"⚠️ Uso de memória alto: {memory_after:.1f}MB")
        
        # Limpeza periódica de cache
        if self.total_predictions % 1000 == 0:
            self._cleanup_cache()
        
        return result
    
    def _cleanup_cache(self):
        """Limpeza periódica de cache para dispositivos IoT"""
        self._feature_cache.clear()
        self._prediction_cache.clear()
        gc.collect()
    
    def get_statistics(self):
        """Obter estatísticas ultra-detalhadas"""
        
        if not self.inference_times:
            return {}
        
        return {
            'model': 'TinyBERT',
            'total_predictions': self.total_predictions,
            'attack_detections': self.attack_detections,
            'attack_rate': self.attack_detections / self.total_predictions if self.total_predictions > 0 else 0,
            'avg_inference_time_ms': np.mean(self.inference_times),
            'max_inference_time_ms': np.max(self.inference_times),
            'min_inference_time_ms': np.min(self.inference_times),
            'std_inference_time_ms': np.std(self.inference_times),
            'p95_inference_time_ms': np.percentile(self.inference_times, 95),
            'p99_inference_time_ms': np.percentile(self.inference_times, 99),
            'throughput_per_second': 1000 / np.mean(self.inference_times),
            'avg_memory_usage_mb': np.mean(self.memory_usage),
            'max_memory_usage_mb': np.max(self.memory_usage),
            'min_memory_usage_mb': np.min(self.memory_usage),
            'avg_cpu_usage_percent': np.mean(self.cpu_usage),
            'max_cpu_usage_percent': np.max(self.cpu_usage),
            'cpu_count': psutil.cpu_count(),
            'system_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'cache_size': len(self._feature_cache)
        }

class UltraLightMonitor:
    def __init__(self, detector, log_file='tinybert_attack_log.json'):
        self.detector = detector
        self.log_file = log_file
        self.data_queue = queue.Queue(maxsize=500)  # Buffer menor para IoT
        self.running = False
        self.stats_interval = 200  # Stats menos frequentes
        self.log_attacks_only = True  # Log apenas ataques para economizar I/O
    
    def log_detection(self, result):
        """Registrar apenas ataques para economizar I/O"""
        
        if not result['is_attack'] and self.log_attacks_only:
            return
            
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                # Log compacto para IoT
                compact_result = {
                    'ts': result['timestamp'][:19],  # Timestamp compacto
                    'cls': result['predicted_class'],
                    'conf': round(result['confidence'], 3),
                    'att': result['is_attack'],
                    'lat': round(result['inference_time_ms'], 2)
                }
                f.write(json.dumps(compact_result) + '\n')
        except Exception as e:
            print(f"❌ Erro ao salvar log: {e}")
    
    def process_data_stream(self):
        """Processar stream ultra-otimizado"""
        
        while self.running:
            try:
                # Timeout menor para responsividade
                features_dict = self.data_queue.get(timeout=0.5)
                
                # Fazer predição
                result = self.detector.predict(features_dict, verbose=False)
                
                # Log e alertas otimizados
                if result['is_attack']:
                    print(f"🚨 ATAQUE: {result['predicted_class']} "
                          f"({result['confidence']:.3f}, {result['inference_time_ms']:.1f}ms)")
                    self.log_detection(result)
                else:
                    # Log muito esparso para tráfego normal
                    if self.detector.total_predictions % 200 == 0:
                        print(f"✅ Normal ({result['confidence']:.3f})")
                
                # Stats menos frequentes
                if self.detector.total_predictions % self.stats_interval == 0:
                    self._show_compact_stats()
                
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Erro: {e}")
    
    def _show_compact_stats(self):
        """Estatísticas compactas para IoT"""
        stats = self.detector.get_statistics()
        print(f"\n📊 TinyBERT: {stats['total_predictions']} pred, "
              f"{stats['attack_detections']} att ({stats['attack_rate']:.1%}), "
              f"{stats['avg_inference_time_ms']:.1f}ms, "
              f"{stats['avg_memory_usage_mb']:.0f}MB")
    
    def start_monitoring(self):
        """Iniciar monitoramento ultra-leve"""
        
        self.running = True
        monitor_thread = threading.Thread(target=self.process_data_stream, daemon=True)
        monitor_thread.start()
        
        print("🎯 Monitoramento TinyBERT iniciado (modo ultra-leve)...")
        return monitor_thread
    
    def stop_monitoring(self):
        """Parar monitoramento"""
        self.running = False
        print("🛑 Monitoramento parado.")
    
    def add_data(self, features_dict):
        """Adicionar dados com descarte inteligente"""
        try:
            self.data_queue.put_nowait(features_dict)
        except queue.Full:
            # Descarte silencioso para IoT
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait(features_dict)
            except queue.Empty:
                pass

def run_ultra_benchmark(detector, num_samples=2000):
    """Benchmark ultra-detalhado para TinyBERT"""
    
    print(f"🏃 Benchmark TinyBERT ultra-otimizado ({num_samples} amostras)...")
    
    # Gerar dados de teste variados
    test_data = []
    for i in range(num_samples):
        # Variar padrões para teste mais realista
        if i % 4 == 0:
            features = {name: np.random.normal(0, 1) for name in detector.feature_names}
        elif i % 4 == 1:
            features = {name: np.random.exponential(1) for name in detector.feature_names}
        elif i % 4 == 2:
            features = {name: np.random.uniform(-2, 2) for name in detector.feature_names}
        else:
            features = {name: 0.0 for name in detector.feature_names}
        test_data.append(features)
    
    # Benchmark com medições detalhadas
    start_time = time.perf_counter()
    memory_start = psutil.Process().memory_info().rss / 1024 / 1024
    
    for i, features in enumerate(test_data):
        detector.predict(features, verbose=False)
        
        if (i + 1) % 200 == 0:
            progress = (i + 1) / num_samples * 100
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"   {progress:.1f}% - Mem: {current_memory:.1f}MB")
    
    total_time = time.perf_counter() - start_time
    memory_end = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Resultados ultra-detalhados
    stats = detector.get_statistics()
    print(f"\n🏆 BENCHMARK TINYBERT COMPLETO")
    print(f"   Amostras: {stats['total_predictions']}")
    print(f"   Tempo total: {total_time:.3f}s")
    print(f"   Latência média: {stats['avg_inference_time_ms']:.3f}ms")
    print(f"   Latência P95/P99: {stats['p95_inference_time_ms']:.3f}/{stats['p99_inference_time_ms']:.3f}ms")
    print(f"   Latência mín/máx: {stats['min_inference_time_ms']:.3f}/{stats['max_inference_time_ms']:.3f}ms")
    print(f"   Throughput: {stats['throughput_per_second']:.1f} pred/s")
    print(f"   Memória inicial/final: {memory_start:.1f}/{memory_end:.1f}MB")
    print(f"   Memória média/máx: {stats['avg_memory_usage_mb']:.1f}/{stats['max_memory_usage_mb']:.1f}MB")
    print(f"   CPU médio/máx: {stats['avg_cpu_usage_percent']:.1f}/{stats['max_cpu_usage_percent']:.1f}%")
    print(f"   Cache size: {stats['cache_size']}")

def simulate_iot_data(csv_file, detector, monitor, delay=0.05):
    """Simular dados IoT em tempo real"""
    
    print(f"📁 Carregando dados IoT: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        # Amostrar para simulação IoT mais realista
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
    except Exception as e:
        print(f"❌ Erro ao carregar: {e}")
        return
    
    print(f"🎬 Simulação IoT: {len(df)} amostras (delay: {delay}s)...")
    
    start_time = time.time()
    
    for idx, row in df.iterrows():
        if not monitor.running:
            break
            
        # Converter linha para dicionário
        features_dict = row.drop('label', errors='ignore').to_dict()
        
        # Adicionar à fila
        monitor.add_data(features_dict)
        
        # Progresso compacto
        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            print(f"📈 {idx + 1}/{len(df)} ({rate:.1f} amostras/s)")
        
        time.sleep(delay)
    
    print("✅ Simulação IoT concluída!")

def main():
    parser = argparse.ArgumentParser(description='TinyBERT Ultra-Otimizado para IoT')
    parser.add_argument('--model', default='tinybert_attack_detector_quantized.onnx', 
                       help='Modelo ONNX do TinyBERT')
    parser.add_argument('--metadata', default='tinybert_metadata.pkl', 
                       help='Metadados do modelo')
    parser.add_argument('--simulate', type=str, 
                       help='Arquivo CSV para simulação IoT')
    parser.add_argument('--delay', type=float, default=0.05, 
                       help='Delay entre amostras (segundos)')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Benchmark ultra-detalhado')
    parser.add_argument('--samples', type=int, default=2000,
                       help='Amostras para benchmark')
    parser.add_argument('--interactive', action='store_true', 
                       help='Modo interativo')
    parser.add_argument('--stress-test', action='store_true',
                       help='Teste de stress para IoT')
    
    args = parser.parse_args()
    
    # Verificações
    if not os.path.exists(args.model):
        print(f"❌ Modelo não encontrado: {args.model}")
        print("💡 Execute TinyBERT_optimization.ipynb primeiro!")
        sys.exit(1)
    
    if not os.path.exists(args.metadata):
        print(f"❌ Metadados não encontrados: {args.metadata}")
        sys.exit(1)
    
    # Inicializar TinyBERT
    try:
        detector = TinyBERTNetworkDetector(args.model, args.metadata)
        monitor = UltraLightMonitor(detector)
    except Exception as e:
        print(f"❌ Erro ao inicializar TinyBERT: {e}")
        sys.exit(1)
    
    if args.benchmark:
        run_ultra_benchmark(detector, args.samples)
        
    elif args.stress_test:
        print("🔥 Teste de stress IoT...")
        run_ultra_benchmark(detector, 10000)
        
    elif args.simulate:
        if not os.path.exists(args.simulate):
            print(f"❌ Arquivo não encontrado: {args.simulate}")
            sys.exit(1)
            
        monitor_thread = monitor.start_monitoring()
        
        try:
            simulate_iot_data(args.simulate, detector, monitor, args.delay)
        except KeyboardInterrupt:
            print("\n🛑 Interrompido")
        finally:
            monitor.stop_monitoring()
            
            # Stats finais compactas
            stats = detector.get_statistics()
            print(f"\n📊 FINAL: {stats['total_predictions']} pred, "
                  f"{stats['attack_detections']} att, "
                  f"{stats['avg_inference_time_ms']:.2f}ms avg, "
                  f"{stats['throughput_per_second']:.0f}/s")
    
    elif args.interactive:
        print("\n🎮 Modo interativo TinyBERT")
        
        while True:
            try:
                input("⏎ Enter para predição ultra-rápida: ")
                
                # Features aleatórias
                test_features = {name: np.random.randn() for name in detector.feature_names}
                result = detector.predict(test_features)
                
                print(f"\n⚡ RESULTADO TINYBERT:")
                print(f"   {result['predicted_class']} ({result['confidence']:.3f})")
                print(f"   {'🚨 ATAQUE' if result['is_attack'] else '✅ NORMAL'}")
                print(f"   {result['inference_time_ms']:.2f}ms, {result['memory_usage_mb']:.0f}MB")
                
            except KeyboardInterrupt:
                break
    
    else:
        print("❓ Use --benchmark, --simulate, --interactive ou --stress-test")
        parser.print_help()

if __name__ == '__main__':
    main() 