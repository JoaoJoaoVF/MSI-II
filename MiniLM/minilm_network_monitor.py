#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Detecção de Ataques de Rede em Tempo Real - MiniLM
Otimizado para Raspberry Pi e dispositivos IoT
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

class MiniLMNetworkDetector:
    def __init__(self, model_path, metadata_path):
        """Inicializar detector de ataques MiniLM"""
        
        print("🚀 Carregando modelo MiniLM...")
        
        # Configurar ONNX Runtime para eficiência máxima
        providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = psutil.cpu_count()
        
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
        
        print(f"✅ Modelo MiniLM carregado com sucesso!")
        print(f"📋 Classes detectáveis: {self.classes}")
        print(f"🔧 Features: {len(self.feature_names)}")
        
        # Estatísticas de performance
        self.total_predictions = 0
        self.attack_detections = 0
        self.inference_times = []
        self.memory_usage = []
        
        # Warm-up do modelo
        self._warmup()
    
    def _warmup(self):
        """Warm-up do modelo para otimizar performance"""
        print("🔥 Aquecendo modelo...")
        dummy_features = {name: np.random.randn() for name in self.feature_names}
        
        # Executar algumas predições para warm-up
        for _ in range(10):
            self.predict(dummy_features, verbose=False)
        
        print("✅ Warm-up concluído!")
    
    def preprocess_features(self, features_dict):
        """Pré-processar features de entrada"""
        
        # Converter para array na ordem correta
        features_array = np.array([features_dict.get(name, 0.0) for name in self.feature_names])
        
        # Normalizar usando o scaler treinado
        features_scaled = self.scaler.transform(features_array.reshape(1, -1))
        
        return features_scaled.astype(np.float32)
    
    def predict(self, features_dict, verbose=True):
        """Fazer predição de ataque"""
        
        # Pré-processar
        features = self.preprocess_features(features_dict)
        
        # Medir uso de memória antes da inferência
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Inferência
        start_time = time.time()
        ort_inputs = {'features': features}
        logits, probabilities = self.session.run(None, ort_inputs)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Medir uso de memória após a inferência
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Processar resultado
        predicted_class_idx = np.argmax(probabilities[0])
        predicted_class = self.classes[predicted_class_idx]
        confidence = probabilities[0][predicted_class_idx]
        
        # Atualizar estatísticas
        self.total_predictions += 1
        self.inference_times.append(inference_time)
        self.memory_usage.append(memory_after)
        
        is_attack = predicted_class.lower() != 'benign'
        if is_attack:
            self.attack_detections += 1
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'model': 'MiniLM',
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'is_attack': is_attack,
            'inference_time_ms': inference_time,
            'memory_usage_mb': memory_after,
            'all_probabilities': probabilities[0].tolist()
        }
        
        if verbose and inference_time > 10:  # Alertar se latência alta
            print(f"⚠️ Latência alta detectada: {inference_time:.2f}ms")
        
        return result
    
    def get_statistics(self):
        """Obter estatísticas detalhadas do detector"""
        
        if not self.inference_times:
            return {}
        
        return {
            'model': 'MiniLM',
            'total_predictions': self.total_predictions,
            'attack_detections': self.attack_detections,
            'attack_rate': self.attack_detections / self.total_predictions if self.total_predictions > 0 else 0,
            'avg_inference_time_ms': np.mean(self.inference_times),
            'max_inference_time_ms': np.max(self.inference_times),
            'min_inference_time_ms': np.min(self.inference_times),
            'std_inference_time_ms': np.std(self.inference_times),
            'throughput_per_second': 1000 / np.mean(self.inference_times),
            'avg_memory_usage_mb': np.mean(self.memory_usage),
            'max_memory_usage_mb': np.max(self.memory_usage),
            'cpu_count': psutil.cpu_count(),
            'system_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024
        }

class RealTimeMonitor:
    def __init__(self, detector, log_file='minilm_attack_log.json'):
        self.detector = detector
        self.log_file = log_file
        self.data_queue = queue.Queue(maxsize=1000)  # Buffer limitado
        self.running = False
        self.stats_interval = 100  # Mostrar stats a cada 100 predições
    
    def log_detection(self, result):
        """Registrar detecção em arquivo"""
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"❌ Erro ao salvar log: {e}")
    
    def process_data_stream(self):
        """Processar stream de dados em tempo real"""
        
        while self.running:
            try:
                # Obter dados da fila
                features_dict = self.data_queue.get(timeout=1)
                
                # Fazer predição
                result = self.detector.predict(features_dict)
                
                # Log e alertas
                if result['is_attack']:
                    print(f"🚨 ATAQUE DETECTADO: {result['predicted_class']} "
                          f"(Confiança: {result['confidence']:.3f}, "
                          f"Latência: {result['inference_time_ms']:.1f}ms)")
                    self.log_detection(result)
                else:
                    if self.detector.total_predictions % 50 == 0:  # Log periódico
                        print(f"✅ Tráfego normal (Confiança: {result['confidence']:.3f})")
                
                # Mostrar estatísticas periodicamente
                if self.detector.total_predictions % self.stats_interval == 0:
                    self._show_stats()
                
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Erro no processamento: {e}")
    
    def _show_stats(self):
        """Mostrar estatísticas em tempo real"""
        stats = self.detector.get_statistics()
        print(f"\n📊 ESTATÍSTICAS (MiniLM)")
        print(f"   Predições: {stats['total_predictions']}")
        print(f"   Ataques: {stats['attack_detections']} ({stats['attack_rate']:.1%})")
        print(f"   Latência: {stats['avg_inference_time_ms']:.1f}ms (±{stats['std_inference_time_ms']:.1f})")
        print(f"   Throughput: {stats['throughput_per_second']:.1f}/s")
        print(f"   Memória: {stats['avg_memory_usage_mb']:.1f}MB")
    
    def start_monitoring(self):
        """Iniciar monitoramento"""
        
        self.running = True
        monitor_thread = threading.Thread(target=self.process_data_stream, daemon=True)
        monitor_thread.start()
        
        print("🎯 Monitoramento MiniLM iniciado...")
        return monitor_thread
    
    def stop_monitoring(self):
        """Parar monitoramento"""
        self.running = False
        print("🛑 Monitoramento parado.")
    
    def add_data(self, features_dict):
        """Adicionar dados para análise"""
        try:
            self.data_queue.put_nowait(features_dict)
        except queue.Full:
            print("⚠️ Buffer cheio, descartando amostra antiga...")
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait(features_dict)
            except queue.Empty:
                pass

def run_benchmark(detector, num_samples=1000):
    """Executar benchmark de performance"""
    
    print(f"🏃 Executando benchmark MiniLM com {num_samples} amostras...")
    
    # Gerar dados de teste
    test_data = []
    for _ in range(num_samples):
        features = {name: np.random.randn() for name in detector.feature_names}
        test_data.append(features)
    
    # Executar benchmark
    start_time = time.time()
    
    for i, features in enumerate(test_data):
        detector.predict(features, verbose=False)
        
        if (i + 1) % 100 == 0:
            progress = (i + 1) / num_samples * 100
            print(f"   Progresso: {progress:.1f}%")
    
    total_time = time.time() - start_time
    
    # Resultados
    stats = detector.get_statistics()
    print(f"\n🏆 RESULTADOS DO BENCHMARK (MiniLM)")
    print(f"   Amostras processadas: {stats['total_predictions']}")
    print(f"   Tempo total: {total_time:.2f}s")
    print(f"   Latência média: {stats['avg_inference_time_ms']:.2f}ms")
    print(f"   Latência mín/máx: {stats['min_inference_time_ms']:.2f}/{stats['max_inference_time_ms']:.2f}ms")
    print(f"   Throughput: {stats['throughput_per_second']:.1f} predições/segundo")
    print(f"   Memória média: {stats['avg_memory_usage_mb']:.1f}MB")
    print(f"   Memória máxima: {stats['max_memory_usage_mb']:.1f}MB")

def simulate_network_data(csv_file, detector, monitor, delay=0.1):
    """Simular dados de rede em tempo real"""
    
    print(f"📁 Carregando dados de simulação: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"❌ Erro ao carregar arquivo: {e}")
        return
    
    print(f"🎬 Iniciando simulação com {len(df)} amostras (delay: {delay}s)...")
    
    for idx, row in df.iterrows():
        if not monitor.running:
            break
            
        # Converter linha para dicionário (excluindo label se existir)
        features_dict = row.drop('label', errors='ignore').to_dict()
        
        # Adicionar à fila de monitoramento
        monitor.add_data(features_dict)
        
        # Mostrar progresso
        if (idx + 1) % 500 == 0:
            progress = (idx + 1) / len(df) * 100
            print(f"📈 Progresso: {progress:.1f}% ({idx + 1}/{len(df)})")
        
        time.sleep(delay)
    
    print("✅ Simulação concluída!")

def main():
    parser = argparse.ArgumentParser(description='Detector de Ataques MiniLM em Tempo Real')
    parser.add_argument('--model', default='minilm_attack_detector_quantized.onnx', 
                       help='Modelo ONNX do MiniLM')
    parser.add_argument('--metadata', default='minilm_metadata.pkl', 
                       help='Metadados do modelo')
    parser.add_argument('--simulate', type=str, 
                       help='Arquivo CSV para simulação')
    parser.add_argument('--delay', type=float, default=0.1, 
                       help='Delay entre amostras (segundos)')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Executar benchmark de performance')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Número de amostras para benchmark')
    parser.add_argument('--interactive', action='store_true', 
                       help='Modo interativo')
    
    args = parser.parse_args()
    
    # Verificar se arquivos existem
    if not os.path.exists(args.model):
        print(f"❌ Modelo não encontrado: {args.model}")
        print("💡 Execute o notebook MiniLM_optimization.ipynb primeiro!")
        sys.exit(1)
    
    if not os.path.exists(args.metadata):
        print(f"❌ Metadados não encontrados: {args.metadata}")
        sys.exit(1)
    
    # Inicializar detector
    try:
        detector = MiniLMNetworkDetector(args.model, args.metadata)
        monitor = RealTimeMonitor(detector)
    except Exception as e:
        print(f"❌ Erro ao inicializar detector MiniLM: {e}")
        sys.exit(1)
    
    if args.benchmark:
        run_benchmark(detector, args.samples)
        
    elif args.simulate:
        if not os.path.exists(args.simulate):
            print(f"❌ Arquivo de simulação não encontrado: {args.simulate}")
            sys.exit(1)
            
        monitor_thread = monitor.start_monitoring()
        
        try:
            simulate_network_data(args.simulate, detector, monitor, args.delay)
        except KeyboardInterrupt:
            print("\n🛑 Interrompido pelo usuário")
        finally:
            monitor.stop_monitoring()
            
            # Mostrar estatísticas finais
            stats = detector.get_statistics()
            print(f"\n📊 ESTATÍSTICAS FINAIS (MiniLM)")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
    
    elif args.interactive:
        print("\n🎮 Modo interativo MiniLM ativado.")
        print("Pressione Enter para gerar predições com dados aleatórios...")
        
        while True:
            try:
                input("⏎ Enter para predição (Ctrl+C para sair): ")
                
                # Gerar features aleatórias
                test_features = {name: np.random.randn() for name in detector.feature_names}
                result = detector.predict(test_features)
                
                print(f"\n🔍 RESULTADO:")
                print(f"   Classe: {result['predicted_class']}")
                print(f"   Confiança: {result['confidence']:.3f}")
                print(f"   É ataque: {'🚨 SIM' if result['is_attack'] else '✅ NÃO'}")
                print(f"   Latência: {result['inference_time_ms']:.2f}ms")
                print(f"   Memória: {result['memory_usage_mb']:.1f}MB")
                
            except KeyboardInterrupt:
                break
    
    else:
        print("❓ Use --simulate, --interactive ou --benchmark")
        parser.print_help()

if __name__ == '__main__':
    main() 