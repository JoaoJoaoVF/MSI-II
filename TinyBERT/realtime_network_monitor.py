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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class NetworkAttackDetector:
    def __init__(self, model_path, metadata_path, confidence_threshold=0.8):
        
        print("Carregando modelo TinyBERT...")
        # Configurar ONNX Runtime para eficiência balanceada
        providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = psutil.cpu_count()  # Usar todos os cores disponíveis
        sess_options.inter_op_num_threads = 2  # Balanceado para workstations
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
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

        # Configurar threshold de confiança
        self.confidence_threshold = confidence_threshold

        print(f"TinyBERT carregado com sucesso!")
        # Mostrar classes como nomes em vez de números
        class_names_list = [self.get_class_name(cls) for cls in self.classes]
        print(f"Classes detectáveis: {class_names_list}")
        print(f"Threshold de confiança: {self.confidence_threshold}")
        
        # Inicializar métricas
        self.total_predictions = 0
        self.attack_detections = 0
        self.attack_types = {}  # Para contagem de tipos de ataque
        self.benign_count = 0   # Contagem de tráfego normal
        self.inference_times = []
        self.cpu_usage = []
        self.memory_usage = []
        
        # Para rastreamento de confiança
        self.high_confidence_predictions = 0
        self.low_confidence_predictions = 0
        
        # Para métricas por tipo de ataque
        self.true_positives = {}  # TP por tipo de ataque
        self.false_positives = {}  # FP por tipo de ataque
        self.false_negatives = {}  # FN por tipo de ataque
        self.true_negatives = {}  # TN por tipo de ataque
        
        # Inicializar contadores para cada tipo de ataque
        for class_name in self.class_names.values():
            self.true_positives[class_name] = 0
            self.false_positives[class_name] = 0
            self.false_negatives[class_name] = 0
            self.true_negatives[class_name] = 0
    
    # Função auxiliar para converter índices de classe para nomes
    def get_class_name(self, class_idx):
        if isinstance(class_idx, (int, np.integer)):
            return self.class_names.get(class_idx, f"Unknown-{class_idx}")
        return class_idx  # Retorna como está se não for um número

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

    def preprocess_features(self, features_dict):
        feature_values = {}
        for feature_name in self.feature_names:
            feature_values[feature_name] = features_dict.get(feature_name, 0.0)
        
        features_df = pd.DataFrame([feature_values])
        
        try:
            features_scaled = self.scaler.transform(features_df)
        except Exception as e:
            print(f"Aviso: Erro na normalização, usando dados sem normalização: {e}")
            features_scaled = features_df.values
        
        return features_scaled.astype(np.float32)
    
    def predict(self, features_dict, verbose=True):
        # Capturar métricas de CPU e memória antes da inferência
        cpu_percent_before = psutil.cpu_percent()
        memory_info_before = psutil.virtual_memory().percent
        
        features = self.preprocess_features(features_dict)
        
        start_time = time.time()
        ort_inputs = {'features': features}
        logits, probabilities = self.session.run(None, ort_inputs)
        inference_time = (time.time() - start_time) * 1000
        
        # Capturar métricas após a inferência
        cpu_percent_after = psutil.cpu_percent()
        memory_info_after = psutil.virtual_memory().percent
        
        # Armazenar métricas
        self.cpu_usage.append(max(cpu_percent_before, cpu_percent_after))
        self.memory_usage.append(max(memory_info_before, memory_info_after))
        
        predicted_class_idx = np.argmax(probabilities[0])
        predicted_class_numeric = self.classes[predicted_class_idx]
        # Converter o índice de classe para nome de classe
        predicted_class = self.get_class_name(predicted_class_numeric)
        confidence = probabilities[0][predicted_class_idx]

        # Fix: Handle both string and integer class representations
        # Fix: Handle both numeric and string class representations
        is_benign = False
        if isinstance(predicted_class_numeric, (int, np.integer)):
            is_benign = (predicted_class_numeric == 2)  # Assuming 2 is BenignTraffic
        else:
            is_benign = predicted_class.lower() in ['benigntraffic', 'benign', 'normal']
            
        is_attack = not is_benign
        
        # Atualizar estatísticas
        if is_attack:
            self.attack_detections += 1
            # Contabilizar tipo de ataque
            self.attack_types[predicted_class] = self.attack_types.get(predicted_class, 0) + 1
        else:
            self.benign_count += 1
            
        # Contabilizar confiança
        if confidence >= self.confidence_threshold:
            self.high_confidence_predictions += 1
        else:
            self.low_confidence_predictions += 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'model': 'TinyBERT',
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'is_attack': is_attack,
            'is_benign': is_benign,
            'inference_time_ms': inference_time,
            'all_probabilities': probabilities[0].tolist()
        }
    
    def get_statistics(self):
        if not self.inference_times:
            return {}
        
        return {
            'model': 'TinyBERT',
            'total_predictions': self.total_predictions,
            'attack_detections': self.attack_detections,
            'benign_detections': self.benign_count,
            'attack_rate': self.attack_detections / self.total_predictions if self.total_predictions > 0 else 0,
            'high_confidence_predictions': self.high_confidence_predictions,
            'low_confidence_predictions': self.low_confidence_predictions,
            'high_confidence_rate': self.high_confidence_predictions / self.total_predictions if self.total_predictions > 0 else 0,
            'attack_types': self.attack_types,
            'avg_inference_time_ms': np.mean(self.inference_times),
            'max_inference_time_ms': np.max(self.inference_times),
            'min_inference_time_ms': np.min(self.inference_times),
            'throughput_per_second': 1000 / np.mean(self.inference_times) if self.inference_times else 0,
            'avg_cpu_usage': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'max_cpu_usage': np.max(self.cpu_usage) if self.cpu_usage else 0,
            'avg_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0,
            'max_memory_usage': np.max(self.memory_usage) if self.memory_usage else 0,
            'confidence_threshold': self.confidence_threshold
        }

class RealTimeMonitor:
    def __init__(self, detector, log_file='attack_log.json', result_file=None):
        self.detector = detector
        self.log_file = log_file
        self.result_file = result_file
        self.data_queue = queue.Queue()
        self.running = False
        self.results = []  
        
        # Para calcular acurácia caso tenhamos rótulos reais
        self.true_labels = []
        self.predicted_labels = [] 
    
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
        """Salva todos os resultados e métricas em um arquivo"""
        if not self.result_file:
            return

        # Calcular métricas gerais
        metrics_by_class = self.detector.get_metrics_by_class()
        
        # Calcular médias das métricas
        avg_precision = np.mean([m['precision'] for m in metrics_by_class.values()])
        avg_recall = np.mean([m['recall'] for m in metrics_by_class.values()])
        avg_f1 = np.mean([m['f1_score'] for m in metrics_by_class.values()])
        avg_accuracy = np.mean([m['accuracy'] for m in metrics_by_class.values()])

        # Preparar resultados
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_predictions': self.detector.total_predictions,
            'attack_detections': self.detector.attack_detections,
            'benign_traffic': self.detector.benign_count,
            'attack_types': self.detector.attack_types,
            'average_inference_time_ms': np.mean(self.detector.inference_times) if self.detector.inference_times else 0,
            'average_cpu_usage': np.mean(self.detector.cpu_usage) if self.detector.cpu_usage else 0,
            'average_memory_usage': np.mean(self.detector.memory_usage) if self.detector.memory_usage else 0,
            'high_confidence_predictions': self.detector.high_confidence_predictions,
            'low_confidence_predictions': self.detector.low_confidence_predictions,
            
            # Métricas médias gerais
            'average_metrics': {
                'precision': float(avg_precision),
                'recall': float(avg_recall),
                'f1_score': float(avg_f1),
                'accuracy': float(avg_accuracy)
            },
            
            # Métricas detalhadas por tipo de ataque
            'metrics_by_attack_type': {
                attack_type: {
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1_score': float(metrics['f1_score']),
                    'accuracy': float(metrics['accuracy']),
                    'confusion_matrix': {
                        'true_positives': int(metrics['true_positives']),
                        'false_positives': int(metrics['false_positives']),
                        'false_negatives': int(metrics['false_negatives']),
                        'true_negatives': int(metrics['true_negatives'])
                    }
                }
                for attack_type, metrics in metrics_by_class.items()
            }
        }

        # Salvar resultados
        with open(self.result_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"\nResultados salvos em: {self.result_file}")
        
        # Imprimir resumo
        print("\nResumo das Métricas:")
        print(f"Precisão Média: {avg_precision:.4f}")
        print(f"Recall Médio: {avg_recall:.4f}")
        print(f"F1-Score Médio: {avg_f1:.4f}")
        print(f"Acurácia Média: {avg_accuracy:.4f}")
        print(f"Total de Predições: {self.detector.total_predictions}")
        print(f"Detecções de Ataque: {self.detector.attack_detections}")
        print(f"Tráfego Normal: {self.detector.benign_count}")
        print(f"Tempo Médio de Inferência: {np.mean(self.detector.inference_times):.2f}ms")
    
    def process_data_stream(self):
        while self.running:
            try:
                # Se temos uma tupla (features, true_label), salvar o rótulo verdadeiro
                data = self.data_queue.get(timeout=1)
                
                if isinstance(data, tuple) and len(data) == 2:
                    features_dict, true_label = data
                    self.true_labels.append(true_label)
                else:
                    features_dict = data
                
                result = self.detector.predict(features_dict)
                self.results.append(result)
                
                # Simplificar mensagens para classificação binária
                if result['is_attack']:
                    message = f"🚨 ATAQUE: {result['predicted_class']} (Confiança: {result['confidence']:.3f})"
                    self.save_result(message)
                    self.log_detection(result)
                else:
                    message = f"✅ TRÁFEGO NORMAL (Confiança: {result['confidence']:.3f})"
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

def simulate_network_data(csv_file, detector, monitor, delay=0.1):  # Delay balanceado
    message = f"Carregando dados de simulação: {csv_file}"
    monitor.save_result(message)
    df = pd.read_csv(csv_file)
    
    message = f"Iniciando simulação TinyBERT com {len(df)} amostras..."
    monitor.save_result(message)
    
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
            monitor.true_labels.append(is_attack)
        else:
            features_dict = row.to_dict()
            monitor.add_data(features_dict)
        
        if (idx + 1) % 100 == 0:
            stats = detector.get_statistics()
            progress_msg = f"\nProcessadas {idx + 1} amostras"
            monitor.save_result(progress_msg)
            monitor.save_result(f"Taxa de ataques: {stats.get('attack_rate', 0):.3f}")
            monitor.save_result(f"Tempo médio: {stats.get('avg_inference_time_ms', 0):.2f} ms")
        
        time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description='Detector de Ataques de Rede TinyBERT - Otimizado para Workstations')
    parser.add_argument('--model', default='tinybert_attack_detector_quantized.onnx', help='Modelo ONNX TinyBERT')
    parser.add_argument('--metadata', default='tinybert_metadata.pkl', help='Metadados do modelo TinyBERT')
    parser.add_argument('--simulate', type=str, help='Arquivo CSV para simulação')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay entre amostras (segundos)')
    parser.add_argument('--interactive', action='store_true', help='Modo interativo')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark de performance')
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
        print("Executando benchmark TinyBERT para workstation...")
        
        test_features = {name: np.random.randn() for name in detector.feature_names}
        
        for i in range(1000):
            detector.predict(test_features)
        
        stats = detector.get_statistics()
        print(f"\nResultados do benchmark TinyBERT:")
        print(f"Predições: {stats['total_predictions']}")
        print(f"Tempo médio: {stats['avg_inference_time_ms']:.2f} ms")
        print(f"Throughput: {stats['throughput_per_second']:.2f} predições/segundo")
        print(f"Uso médio de CPU: {stats['avg_cpu_usage']:.2f}%")
        print(f"Uso médio de memória: {stats['avg_memory_usage']:.2f}%")
        
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
                
            except KeyboardInterrupt:
                break
    
    else:
        print("Use --simulate, --interactive ou --benchmark")
        parser.print_help()

if __name__ == '__main__':
    main()
