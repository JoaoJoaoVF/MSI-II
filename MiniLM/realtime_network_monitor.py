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

warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class NetworkAttackDetector:
    def __init__(self, model_path, metadata_path, confidence_threshold=0.8):
        
        print("Carregando modelo...")
        self.session = ort.InferenceSession(model_path)
        
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
        
        print(f"Modelo carregado com sucesso!")
        print(f"Classes detectáveis: {self.classes}")
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
    
    def preprocess_features(self, features_dict):
        # [manter código existente]
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
        predicted_class = self.classes[predicted_class_idx]
        confidence = probabilities[0][predicted_class_idx]
        
        self.total_predictions += 1
        self.inference_times.append(inference_time)

        # Simplificar a classificação para apenas "ataque" ou "normal"
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
        
        # Calcular métricas por classe
        metrics_by_class = self.get_metrics_by_class()
        
        # Calcular médias das métricas
        avg_precision = np.mean([m['precision'] for m in metrics_by_class.values()])
        avg_recall = np.mean([m['recall'] for m in metrics_by_class.values()])
        avg_f1 = np.mean([m['f1_score'] for m in metrics_by_class.values()])
        avg_accuracy = np.mean([m['accuracy'] for m in metrics_by_class.values()])
        
        return {
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
            'confidence_threshold': self.confidence_threshold,
            
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
        if self.result_file and self.results:
            # Obter estatísticas do detector
            stats = self.detector.get_statistics()
            metrics_by_class = stats.get('metrics_by_attack_type', {})
            avg_metrics = stats.get('average_metrics', {})
            
            with open(self.result_file, 'w', encoding='utf-8') as f:
                f.write("=== RESULTADOS DA ANÁLISE ===\n")
                f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total de amostras processadas: {len(self.results)}\n")
                f.write(f"Threshold de confiança: {self.detector.confidence_threshold}\n\n")
                
                # Estatísticas básicas
                attacks = [r for r in self.results if r['is_attack']]
                benign = [r for r in self.results if not r['is_attack']]
                
                f.write(f"Ataques detectados: {len(attacks)}\n")
                f.write(f"Tráfego normal: {len(benign)}\n")
                f.write(f"Taxa de ataques: {len(attacks)/len(self.results)*100:.2f}%\n\n")
                
                # Estatísticas de confiança
                confidences = [r['confidence'] for r in self.results]
                low_confidence = [r for r in self.results if r['confidence'] < self.detector.confidence_threshold]
                high_confidence = [r for r in self.results if r['confidence'] >= self.detector.confidence_threshold]
                
                f.write("=== ESTATÍSTICAS DE CONFIANÇA ===\n")
                f.write(f"Predições com alta confiança: {len(high_confidence)} ({len(high_confidence)/len(self.results)*100:.2f}%)\n")
                f.write(f"Predições com baixa confiança: {len(low_confidence)} ({len(low_confidence)/len(self.results)*100:.2f}%)\n")
                f.write(f"Confiança média: {np.mean(confidences):.3f}\n")
                f.write(f"Confiança mínima: {min(confidences):.3f}\n")
                f.write(f"Confiança máxima: {max(confidences):.3f}\n\n")
                
                # Métricas médias
                f.write("=== MÉTRICAS MÉDIAS DE AVALIAÇÃO ===\n")
                f.write(f"Precisão média: {avg_metrics.get('precision', 0):.4f}\n")
                f.write(f"Recall médio: {avg_metrics.get('recall', 0):.4f}\n")
                f.write(f"F1-Score médio: {avg_metrics.get('f1_score', 0):.4f}\n")
                f.write(f"Acurácia média: {avg_metrics.get('accuracy', 0):.4f}\n\n")
                
                # Métricas por tipo de ataque
                f.write("=== MÉTRICAS POR TIPO DE ATAQUE ===\n")
                for attack_type, metrics in metrics_by_class.items():
                    f.write(f"\n{attack_type}:\n")
                    f.write(f"  Precisão: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
                    f.write(f"  Acurácia: {metrics['accuracy']:.4f}\n")
                    f.write(f"  Verdadeiros Positivos: {metrics['true_positives']}\n")
                    f.write(f"  Falsos Positivos: {metrics['false_positives']}\n")
                    f.write(f"  Falsos Negativos: {metrics['false_negatives']}\n")
                    f.write(f"  Verdadeiros Negativos: {metrics['true_negatives']}\n")
                
                # Estatísticas de performance
                f.write("\n=== ESTATÍSTICAS DE PERFORMANCE ===\n")
                f.write(f"Tempo médio de inferência: {stats['avg_inference_time_ms']:.2f} ms\n")
                f.write(f"Throughput: {stats['throughput_per_second']:.2f} inferências/segundo\n")
                f.write(f"Uso médio de CPU: {stats['avg_cpu_usage']:.1f}%\n")
                f.write(f"Uso médio de memória: {stats['avg_memory_usage']:.1f}%\n")
                
            print(f"\nResultados salvos em: {self.result_file}")
    
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
        
        print("Monitoramento iniciado...")
        return monitor_thread
    
    def stop_monitoring(self):
        self.running = False
    
    def add_data(self, features_dict):
        self.data_queue.put(features_dict)

def simulate_network_data(csv_file, detector, monitor, delay=1.0):
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
    parser = argparse.ArgumentParser(description='Detector de Ataques de Rede MiniLM - Otimizado para Workstations')
    parser.add_argument('--model', default='minilm_attack_detector_quantized.onnx', help='Modelo ONNX MiniLM')
    parser.add_argument('--metadata', default='minilm_metadata.pkl', help='Metadados do modelo MiniLM')
    parser.add_argument('--simulate', type=str, help='Arquivo CSV para simulação')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay entre amostras (segundos)')
    parser.add_argument('--interactive', action='store_true', help='Modo interativo')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark de performance')
    parser.add_argument('--output', type=str, help='Arquivo de saída personalizado')
    
    args = parser.parse_args()
    
    result_file = None
    if args.simulate:
        csv_basename = os.path.splitext(os.path.basename(args.simulate))[0]
        result_file = f"result-minilm-part-{csv_basename}.txt"
        print(f"Resultados serão salvos em: {result_file}")
    elif args.output:
        result_file = args.output
        print(f"Resultados serão salvos em: {result_file}")
    
    try:
        detector = NetworkAttackDetector(args.model, args.metadata)
        monitor = RealTimeMonitor(detector, result_file=result_file)
    except Exception as e:
        print(f"Erro ao inicializar detector: {e}")
        sys.exit(1)
    
    if args.benchmark:
        print("Executando benchmark...")
        
        test_features = {name: np.random.randn() for name in detector.feature_names}
        
        for i in range(1000):
            detector.predict(test_features)
        



        stats = detector.get_statistics()
        print(f"\nResultados do benchmark:")
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
                print(f"\n✅ Análise concluída! Resultados salvos em: {result_file}")
    
    elif args.interactive:
        print("\nModo interativo ativado.")
        print("Digite valores para as features ou 'sair' para encerrar.")
        print(f"Features necessárias: {detector.feature_names[:5]}... (total: {len(detector.feature_names)})")
        
        while True:
            try:
                input("Pressione Enter para gerar predição com dados aleatórios (ou Ctrl+C para sair): ")
                
                test_features = {name: np.random.randn() for name in detector.feature_names}
                result = detector.predict(test_features)
                
                print(f"\nResultado:")
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
