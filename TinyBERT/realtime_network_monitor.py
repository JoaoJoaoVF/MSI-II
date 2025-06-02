#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import onnxruntime as ort
import numpy as np
import pandas as pd
import pickle
import time
import argparse
import json
import os
import warnings
from datetime import datetime
import threading
import queue
import sys
import psutil
import gc

warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class NetworkAttackDetector:
    def __init__(self, model_path, metadata_path, confidence_threshold=0.8):
        
        print("Carregando modelo TinyBERT otimizado...")
        
        # Configurar ONNX Runtime para máxima eficiência em IoT
        providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = min(psutil.cpu_count(), 2)  # Limitar threads para IoT
        sess_options.inter_op_num_threads = 1  # Single thread para economia de recursos
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        self.session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=providers
        )
        
        print("Carregando metadados...")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.scaler = self.metadata['scaler']
        self.label_encoder = self.metadata['label_encoder']
        self.feature_names = self.metadata['feature_names']
        self.classes = self.metadata['classes']
        
        # Configurar threshold de confiança
        self.confidence_threshold = confidence_threshold
        
        # Definir classes consideradas menos perigosas (podem ser ajustadas)
        self.low_threat_classes = [
            'VulnerabilityScan',  # Menos crítico que DDoS
            'Recon-PingSweep',    # Reconhecimento básico
            'BrowserHijacking'    # Menos impactante que DDoS
        ]
        
        print(f"TinyBERT carregado com sucesso!")
        print(f"Classes detectáveis: {self.classes}")
        print(f"Threshold de confiança: {self.confidence_threshold}")
        print(f"Classes de baixa ameaça: {self.low_threat_classes}")
        print(f"Uso de memória inicial: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")
        print(f"CPUs disponíveis: {psutil.cpu_count()}")
        
        self.total_predictions = 0
        self.attack_detections = 0
        self.inference_times = []
        self.memory_usage = []
        self.cpu_usage = []                 # inicializar antes de usar em predict()

        # Cache para otimização IoT
        self._feature_cache = {}
        
        # Warm-up ultra-rápido
        self._warmup()
    
    def _warmup(self):
        """Warm-up mínimo para dispositivos IoT"""
        print("Aquecendo TinyBERT...")
        dummy_features = {name: 0.0 for name in self.feature_names}
        
        # Apenas 3 iterações para economia de recursos
        for _ in range(3):
            self.predict(dummy_features, verbose=False)
        
        gc.collect()
        print("Warm-up concluído!")
    
    def preprocess_features(self, features_dict):
        
        # Cache otimizado para IoT
        cache_key = hash(tuple(sorted(features_dict.items())))
        
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        feature_values = {}
        for feature_name in self.feature_names:
            feature_values[feature_name] = features_dict.get(feature_name, 0.0)
        
        features_df = pd.DataFrame([feature_values])
        
        try:
            features_scaled = self.scaler.transform(features_df)
        except Exception as e:
            print(f"Aviso: Erro na normalização, usando dados sem normalização: {e}")
            features_scaled = features_df.values
        
        features_scaled = features_scaled.astype(np.float32)
        
        # Cache limitado para IoT (máximo 50 entradas)
        if len(self._feature_cache) < 50:
            self._feature_cache[cache_key] = features_scaled
        
        return features_scaled
    
    def predict(self, features_dict, verbose=True):
        
        # Medir recursos antes da inferência
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        cpu_before = process.cpu_percent()
        
        features = self.preprocess_features(features_dict)
        
        start_time = time.perf_counter()  # Maior precisão para IoT
        ort_inputs = {'features': features}
        logits, probabilities = self.session.run(None, ort_inputs)
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Medir recursos após a inferência
        memory_after = process.memory_info().rss / 1024 / 1024
        cpu_after = process.cpu_percent()
        
        predicted_class_idx = np.argmax(probabilities[0])
        raw_label = self.classes[predicted_class_idx]
        
        # Converter raw_label (normalmente int) de volta para string
        try:
            # Se label_encoder for um sklearn.preprocessing.LabelEncoder, faz inverse_transform
            predicted_class = self.label_encoder.inverse_transform([raw_label])[0]
        except Exception:
            # Se falhar, converte em str
            predicted_class = str(raw_label)
        
        confidence = probabilities[0][predicted_class_idx]
        
        self.total_predictions += 1
        self.inference_times.append(inference_time)
        self.memory_usage.append(memory_after)
        self.cpu_usage.append(cpu_after)

        # Lógica robusta para determinar ataques - verificar tráfego benigno primeiro
        is_benign = predicted_class.lower() in ['benigntraffic', 'benign', 'normal']
        
        if is_benign:
            # Tráfego benigno nunca é considerado ataque
            is_critical_attack = False
        else:
            high_confidence_attack = confidence > self.confidence_threshold
            is_high_threat = predicted_class not in self.low_threat_classes
            
            ddos_classes = [cls for cls in self.classes if isinstance(cls, str) and ('DDoS' in cls or 'DoS' in cls)]
            is_ddos = predicted_class in ddos_classes
            
            is_critical_attack = (
                (high_confidence_attack and is_high_threat) or
                (high_confidence_attack and is_ddos) or
                (confidence > 0.9)
            )
        
        if is_critical_attack:
            self.attack_detections += 1
        
        # Alertas específicos para TinyBERT/IoT
        if verbose:
            if inference_time > 5:  # Alerta para TinyBERT se > 5ms
                print(f"⚠️ Latência alta para TinyBERT: {inference_time:.2f}ms")
            if memory_after > 200:  # Alerta se > 200MB para IoT
                print(f"⚠️ Uso de memória alto para IoT: {memory_after:.1f}MB")
            if cpu_after > 80:  # Alerta se CPU > 80%
                print(f"⚠️ Uso de CPU alto: {cpu_after:.1f}%")
        
        # Limpeza periódica de cache para IoT
        if self.total_predictions % 500 == 0:
            self._cleanup_cache()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'model': 'TinyBERT',
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'is_attack': is_critical_attack,
            'is_benign': is_benign,
            'is_high_threat': (predicted_class not in self.low_threat_classes) if not is_benign else False,
            'is_ddos': is_ddos if not is_benign else False,
            'confidence_threshold': self.confidence_threshold,
            'inference_time_ms': inference_time,
            'memory_usage_mb': memory_after,
            'cpu_usage_percent': cpu_after,
            'all_probabilities': probabilities[0].tolist()
        }
    
    def _cleanup_cache(self):
        """Limpeza de cache balanceada"""
        cache_items = list(self._feature_cache.items())
        half_size = len(cache_items) // 2
        self._feature_cache = dict(cache_items[-half_size:])
        gc.collect()
    
    def get_statistics(self):
        
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
            'throughput_per_second': 1000 / np.mean(self.inference_times) if self.inference_times else 0,
            'avg_memory_usage_mb': np.mean(self.memory_usage),
            'max_memory_usage_mb': np.max(self.memory_usage),
            'min_memory_usage_mb': np.min(self.memory_usage),
            'avg_cpu_usage_percent': np.mean(self.cpu_usage),
            'max_cpu_usage_percent': np.max(self.cpu_usage),
            'confidence_threshold': self.confidence_threshold,
            'cache_size': len(self._feature_cache),
            'cpu_count': psutil.cpu_count(),
            'system_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024
        }

class RealTimeMonitor:
    def __init__(self, detector, log_file='tinybert_attack_log.json', result_file=None):
        self.detector = detector
        self.log_file = log_file
        self.result_file = result_file
        self.data_queue = queue.Queue(maxsize=500)  # Buffer menor para IoT
        self.running = False
        self.results = []  
    
    def log_detection(self, result):
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
    
    def save_result(self, message):
        
        if self.result_file:
            with open(self.result_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        else:
            print(message)
    
    def save_all_results(self):
        
        if self.result_file and self.results:
            with open(self.result_file, 'w', encoding='utf-8') as f:
                f.write("=== RESULTADOS DA ANÁLISE TinyBERT ===\n")
                f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Modelo: TinyBERT (Otimizado para IoT)\n")
                f.write(f"Total de amostras processadas: {len(self.results)}\n")
                f.write(f"Threshold de confiança: {self.detector.confidence_threshold}\n\n")
                
                attacks = [r for r in self.results if r['is_attack']]
                low_confidence = [r for r in self.results if r['confidence'] < self.detector.confidence_threshold]
                high_confidence = [r for r in self.results if r['confidence'] >= self.detector.confidence_threshold]
                
                f.write(f"Ataques críticos detectados: {len(attacks)}\n")
                f.write(f"Taxa de ataques críticos: {len(attacks)/len(self.results)*100:.2f}%\n")
                f.write(f"Atividade normal/baixo risco: {len(self.results) - len(attacks)}\n")
                f.write(f"Predições baixa confiança: {len(low_confidence)}\n")
                f.write(f"Predições alta confiança: {len(high_confidence)}\n\n")
                
                # Performance específica para workstation
                inference_times = [r['inference_time_ms'] for r in self.results]
                memory_usage = [r.get('memory_usage_mb', 0) for r in self.results]
                cpu_usage = [r.get('cpu_usage_percent', 0) for r in self.results]
                
                f.write("=== PERFORMANCE WORKSTATION ===\n")
                f.write(f"Tempo médio de inferência: {np.mean(inference_times):.2f}ms\n")
                f.write(f"Tempo máximo de inferência: {max(inference_times):.2f}ms\n")
                f.write(f"Desvio padrão de inferência: {np.std(inference_times):.2f}ms\n")
                f.write(f"P95 de inferência: {np.percentile(inference_times, 95):.2f}ms\n")
                f.write(f"P99 de inferência: {np.percentile(inference_times, 99):.2f}ms\n")
                f.write(f"Throughput médio: {1000/np.mean(inference_times):.1f} predições/segundo\n")
                f.write(f"Uso médio de memória: {np.mean(memory_usage):.1f}MB\n")
                f.write(f"Uso máximo de memória: {max(memory_usage):.1f}MB\n")
                f.write(f"Uso médio de CPU: {np.mean(cpu_usage):.1f}%\n")
                f.write(f"Uso máximo de CPU: {max(cpu_usage):.1f}%\n\n")
                
                if attacks:
                    attack_types = {}
                    for attack in attacks:
                        attack_type = attack['predicted_class']
                        attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
                    
                    f.write("=== TIPOS DE ATAQUES CRÍTICOS DETECTADOS ===\n")
                    for attack_type, count in sorted(attack_types.items()):
                        f.write(f"{attack_type}: {count} ocorrências\n")
                    f.write("\n")
                
                # Analisar distribuição de confiança
                confidence_levels = [r['confidence'] for r in self.results]
                f.write("=== DISTRIBUIÇÃO DE CONFIANÇA ===\n")
                f.write(f"Confiança média: {np.mean(confidence_levels):.3f}\n")
                f.write(f"Confiança mediana: {np.median(confidence_levels):.3f}\n")
                f.write(f"Confiança mínima: {min(confidence_levels):.3f}\n")
                f.write(f"Confiança máxima: {max(confidence_levels):.3f}\n")
                f.write(f"Desvio padrão confiança: {np.std(confidence_levels):.3f}\n\n")
                
                f.write("=== DETALHES DAS DETECÇÕES ===\n")
                for i, result in enumerate(self.results, 1):
                    if result['is_attack']:
                        status = "🚨 ATAQUE CRÍTICO"
                    elif result.get('is_benign', False):
                        status = "✅ TRÁFEGO NORMAL"
                    elif result['confidence'] >= self.detector.confidence_threshold:
                        status = "⚠️ ATIVIDADE SUSPEITA"
                    else:
                        status = "🔍 BAIXO RISCO"
                        
                    f.write(f"Amostra {i}: {status}\n")
                    f.write(f"  Classe: {result['predicted_class']}\n")
                    f.write(f"  Confiança: {result['confidence']:.3f}\n")
                    f.write(f"  É tráfego benigno: {result.get('is_benign', 'N/A')}\n")
                    f.write(f"  É DDoS: {result.get('is_ddos', 'N/A')}\n")
                    f.write(f"  Alta ameaça: {result.get('is_high_threat', 'N/A')}\n")
                    f.write(f"  Tempo de inferência: {result['inference_time_ms']:.2f} ms\n")
                    f.write(f"  Uso de memória: {result.get('memory_usage_mb', 'N/A')} MB\n")
                    f.write(f"  Uso de CPU: {result.get('cpu_usage_percent', 'N/A')}%\n")
                    f.write(f"  Timestamp: {result['timestamp']}\n")
                    f.write("\n")
    
    def process_data_stream(self):
        
        while self.running:
            try:
                features_dict = self.data_queue.get(timeout=1)
                
                result = self.detector.predict(features_dict)
                
                self.results.append(result)
                
                if result['is_attack']:
                    message = f"🚨 ATAQUE CRÍTICO: {result['predicted_class']} (Confiança: {result['confidence']:.3f}, Latência: {result['inference_time_ms']:.1f}ms)"
                    self.save_result(message)
                    self.log_detection(result)
                elif result.get('is_benign', False):
                    message = f"✅ TRÁFEGO NORMAL: {result['predicted_class']} (Confiança: {result['confidence']:.3f})"
                    self.save_result(message)
                elif result['confidence'] >= self.detector.confidence_threshold:
                    message = f"⚠️ ATIVIDADE SUSPEITA: {result['predicted_class']} (Confiança: {result['confidence']:.3f})"
                    self.save_result(message)
                else:
                    # Para IoT, log menos frequente para economizar recursos
                    if self.detector.total_predictions % 100 == 0:
                        message = f"🔍 BAIXO RISCO: {result['predicted_class']} (Confiança: {result['confidence']:.3f})"
                        self.save_result(message)
                
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                error_msg = f"Erro no processamento: {e}"
                self.save_result(error_msg)
    
    def start_monitoring(self):
        
        self.running = True
        monitor_thread = threading.Thread(target=self.process_data_stream)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print("Monitoramento TinyBERT iniciado...")
        return monitor_thread
    
    def stop_monitoring(self):
        self.running = False
    
    def add_data(self, features_dict):
        self.data_queue.put(features_dict)

def simulate_network_data(csv_file, detector, monitor, delay=0.05):  # Delay menor para IoT
    message = f"Carregando dados de simulação: {csv_file}"
    monitor.save_result(message)
    df = pd.read_csv(csv_file)
    
    message = f"Iniciando simulação TinyBERT com {len(df)} amostras..."
    monitor.save_result(message)
    
    for idx, row in df.iterrows():
        # drop label sem erro, caso não exista
        features_dict = row.drop(labels=['label'], errors='ignore').to_dict()
        
        monitor.add_data(features_dict)
        
        if (idx + 1) % 200 == 0:  # Stats menos frequentes para IoT
            stats = detector.get_statistics()
            progress_msg = f"\nProcessadas {idx + 1} amostras"
            monitor.save_result(progress_msg)
            monitor.save_result(f"Taxa de ataques: {stats.get('attack_rate', 0):.3f}")
            monitor.save_result(f"Tempo médio: {stats.get('avg_inference_time_ms', 0):.2f} ms")
            monitor.save_result(f"Memória média: {stats.get('avg_memory_usage_mb', 0):.1f} MB")
            monitor.save_result(f"CPU médio: {stats.get('avg_cpu_usage_percent', 0):.1f}%")
        
        time.sleep(delay)

def main():
    parser = argparse.ArgumentParser(description='Detector de Ataques de Rede TinyBERT - Otimizado para IoT')
    parser.add_argument('--model', default='tinybert_attack_detector_quantized.onnx', help='Modelo ONNX TinyBERT')
    parser.add_argument('--metadata', default='tinybert_metadata.pkl', help='Metadados do modelo TinyBERT')
    parser.add_argument('--simulate', type=str, help='Arquivo CSV para simulação')
    parser.add_argument('--delay', type=float, default=0.05, help='Delay entre amostras (segundos)')
    parser.add_argument('--interactive', action='store_true', help='Modo interativo')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark de performance IoT')
    parser.add_argument('--output', type=str, help='Arquivo de saída personalizado')
    
    args = parser.parse_args()
    
    result_file = None
    if args.simulate:
        csv_basename = os.path.splitext(os.path.basename(args.simulate))[0]
        result_file = f"result-tinybert-part-{csv_basename}.txt"
        print(f"Resultados TinyBERT serão salvos em: {result_file}")
    elif args.output:
        result_file = args.output
        print(f"Resultados serão salvos em: {result_file}")
    
    try:
        detector = NetworkAttackDetector(args.model, args.metadata)
        monitor = RealTimeMonitor(detector, result_file=result_file)
    except Exception as e:
        print(f"Erro ao inicializar detector TinyBERT: {e}")
        sys.exit(1)
    
    if args.benchmark:
        print("Executando benchmark TinyBERT para IoT...")
        
        test_features = {name: np.random.randn() for name in detector.feature_names}
        
        for i in range(1000):
            detector.predict(test_features, verbose=False)
        
        stats = detector.get_statistics()
        print(f"\nResultados do benchmark TinyBERT:")
        print(f"Predições: {stats['total_predictions']}")
        print(f"Tempo médio: {stats['avg_inference_time_ms']:.2f} ms")
        print(f"Tempo P95: {stats['p95_inference_time_ms']:.2f} ms")
        print(f"Tempo P99: {stats['p99_inference_time_ms']:.2f} ms")
        print(f"Throughput: {stats['throughput_per_second']:.2f} predições/segundo")
        print(f"Memória média: {stats['avg_memory_usage_mb']:.1f} MB")
        print(f"CPU médio: {stats['avg_cpu_usage_percent']:.1f}%")
        print(f"Tamanho do cache: {stats['cache_size']}")
        
    elif args.simulate:
        monitor_thread = monitor.start_monitoring()
        
        try:
            simulate_network_data(args.simulate, detector, monitor, args.delay)
            
            monitor.data_queue.join()
            
        except KeyboardInterrupt:
            monitor.save_result("Interrompido pelo usuário")
        finally:
            monitor.stop_monitoring()
            
            monitor.save_all_results()
            
            stats = detector.get_statistics()
            final_stats = f"\n=== ESTATÍSTICAS FINAIS TinyBERT ==="
            monitor.save_result(final_stats)
            for key, value in stats.items():
                monitor.save_result(f"{key}: {value}")
            
            if result_file:
                print(f"\n✅ Análise TinyBERT concluída! Resultados salvos em: {result_file}")
    
    elif args.interactive:
        print("\nModo interativo TinyBERT ativado.")
        print("Digite valores para as features ou 'sair' para encerrar.")
        print(f"Features necessárias: {detector.feature_names[:5]}... (total: {len(detector.feature_names)})")
        
        while True:
            try:
                input("Pressione Enter para gerar predição com dados aleatórios (ou Ctrl+C para sair): ")
                
                test_features = {name: np.random.randn() for name in detector.feature_names}
                result = detector.predict(test_features)
                
                print(f"\nResultado TinyBERT:")
                print(f"  Classe: {result['predicted_class']}")
                print(f"  Confiança: {result['confidence']:.3f}")
                print(f"  É ataque: {result['is_attack']}")
                print(f"  Tempo: {result['inference_time_ms']:.2f} ms")
                print(f"  Memória: {result['memory_usage_mb']:.1f} MB")
                print(f"  CPU: {result['cpu_usage_percent']:.1f}%")
                
            except KeyboardInterrupt:
                break
    
    else:
        print("Use --simulate, --interactive ou --benchmark")
        parser.print_help()

if __name__ == '__main__':
    main()
