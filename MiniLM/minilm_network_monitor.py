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
        
        # Mapeamento de classes numéricas para nomes
        self.class_names = {
            1: "Backdoor_Malware",
            2: "BenignTraffic",
            3: "BrowserHijacking",
            4: "CommandInjection",
            5: "DDoS-ACK_Fragmentation",
            6: "DDoS-HTTP_Flood",
            7: "DDoS-ICMP_Flood",
            8: "DDoS-ICMP_Fragmentation",
            9: "DDoS-PSHACK_Flood",
            10: "DDoS-RSTFINFlood",
            11: "DDoS-SYN_Flood",
            12: "DDoS-SlowLoris",
            13: "DDoS-SynonymousIP_Flood",
            14: "DDoS-TCP_Flood",
            15: "DDoS-UDP_Flood",
            16: "DDoS-UDP_Fragmentation",
            17: "DNS_Spoofing",
            18: "DictionaryBruteForce",
            19: "DoS-HTTP_Flood",
            20: "DoS-SYN_Flood",
            21: "DoS-TCP_Flood",
            22: "DoS-UDP_Flood",
            23: "MITM-ArpSpoofing",
            24: "Mirai-greeth_flood",
            25: "Mirai-greip_flood",
            26: "Mirai-udpplain",
            27: "Recon-HostDiscovery",
            28: "Recon-OSScan",
            29: "Recon-PingSweep",
            30: "Recon-PortScan",
            31: "SqlInjection",
            32: "Uploading_Attack",
            33: "VulnerabilityScan",
            34: "XSS"
        }
        
        print(f"✅ Modelo MiniLM carregado com sucesso!")
        print(f"📋 Classes detectáveis: {self.classes}")
        print(f"🔧 Features: {len(self.feature_names)}")
        
        # Estatísticas de performance
        self.total_predictions = 0
        self.attack_detections = 0
        self.inference_times = []
        self.memory_usage = []
        
        # Métricas por tipo de ataque
        self.true_positives = {}
        self.false_positives = {}
        self.false_negatives = {}
        self.true_negatives = {}
        
        # Inicializar contadores para cada tipo de ataque
        for class_name in self.class_names.values():
            self.true_positives[class_name] = 0
            self.false_positives[class_name] = 0
            self.false_negatives[class_name] = 0
            self.true_negatives[class_name] = 0
        
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
        
        # Calcular métricas por classe
        metrics_by_class = self.get_metrics_by_class()
        
        # Calcular médias das métricas
        avg_precision = np.mean([m['precision'] for m in metrics_by_class.values()])
        avg_recall = np.mean([m['recall'] for m in metrics_by_class.values()])
        avg_f1 = np.mean([m['f1_score'] for m in metrics_by_class.values()])
        avg_accuracy = np.mean([m['accuracy'] for m in metrics_by_class.values()])
        
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
            'system_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            
            # Métricas médias
            'average_metrics': {
                'precision': float(avg_precision),
                'recall': float(avg_recall),
                'f1_score': float(avg_f1),
                'accuracy': float(avg_accuracy)
            },
            
            # Métricas detalhadas por tipo de ataque
            'metrics_by_attack_type': metrics_by_class
        }

    def get_class_name(self, class_idx):
        """Converter índice de classe para nome"""
        if isinstance(class_idx, (int, np.integer)):
            return self.class_names.get(class_idx, f"Unknown-{class_idx}")
        return class_idx

    def update_metrics(self, predicted_class, true_class):
        """Atualiza as métricas de classificação para cada tipo de ataque"""
        predicted_name = self.get_class_name(predicted_class)
        true_name = self.get_class_name(true_class)
        
        for class_name in self.class_names.values():
            if class_name == predicted_name and class_name == true_name:
                self.true_positives[class_name] += 1
            elif class_name == predicted_name and class_name != true_name:
                self.false_positives[class_name] += 1
            elif class_name != predicted_name and class_name == true_name:
                self.false_negatives[class_name] += 1
            else:
                self.true_negatives[class_name] += 1

    def get_metrics_by_class(self):
        """Calcula métricas para cada tipo de ataque"""
        metrics = {}
        
        for class_name in self.class_names.values():
            tp = self.true_positives[class_name]
            fp = self.false_positives[class_name]
            fn = self.false_negatives[class_name]
            tn = self.true_negatives[class_name]
            
            # Evitar divisão por zero
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn
            }
        
        return metrics

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
    """Simular stream de dados de rede usando arquivo CSV"""
    
    print(f"📂 Carregando dados: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"🚀 Iniciando simulação com {len(df)} amostras...")
    
    has_labels = 'label' in df.columns
    
    for idx, row in df.iterrows():
        if has_labels:
            # Determinar se o rótulo indica ataque (1) ou normal (0)
            label_value = row['label']
            
            # Fix: Handle both string and integer labels
            is_attack = 1
            if isinstance(label_value, str):
                if label_value.lower() in ['benigntraffic', 'benign', 'normal']:
                    is_attack = 0
            elif label_value == 0:  # Assume 0 is benign traffic if numeric
                is_attack = 0
            
            features_dict = row.drop('label').to_dict()
            result = detector.predict(features_dict)
            
            # Atualizar métricas por tipo de ataque
            predicted_class = result['predicted_class']
            true_class = 'BenignTraffic' if not is_attack else predicted_class
            detector.update_metrics(predicted_class, true_class)
            
            monitor.add_data((features_dict, is_attack))
        else:
            features_dict = row.to_dict()
            monitor.add_data(features_dict)
        
        # Mostrar progresso e métricas periodicamente
        if (idx + 1) % 100 == 0:
            stats = detector.get_statistics()
            print(f"\n📊 Processadas {idx + 1} amostras")
            print(f"🎯 Taxa de ataques: {stats['attack_rate']:.3f}")
            print(f"⚡ Tempo médio: {stats['avg_inference_time_ms']:.2f}ms")
            
            # Mostrar métricas de avaliação
            metrics = stats['average_metrics']
            print("\n📈 Métricas de Avaliação:")
            print(f"Precisão: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"Acurácia: {metrics['accuracy']:.4f}")
        
        time.sleep(delay)

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