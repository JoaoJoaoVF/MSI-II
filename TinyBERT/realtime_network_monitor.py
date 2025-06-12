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
    
    # Função auxiliar para converter índices de classe para nomes
    def get_class_name(self, class_idx):
        if isinstance(class_idx, (int, np.integer)):
            return self.class_names.get(class_idx, f"Unknown-{class_idx}")
        return class_idx  # Retorna como está se não for um número

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
        if self.result_file and self.results:
            with open(self.result_file, 'w', encoding='utf-8') as f:
                f.write("=== RESULTADOS DA ANÁLISE TinyBERT ===\n")
                f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Modelo: TinyBERT (Otimizado para Workstations)\n")
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
                f.write(f"Confiança média: {np.mean(confidences):.4f}\n")
                f.write(f"Confiança mediana: {np.median(confidences):.4f}\n")
                f.write(f"Confiança mínima: {np.min(confidences):.4f}\n")
                f.write(f"Confiança máxima: {np.max(confidences):.4f}\n")
                f.write(f"Desvio padrão da confiança: {np.std(confidences):.4f}\n\n")
                
                # Lista de todos os ataques e suas incidências
                attack_types = {}
                for result in self.results:
                    if result['is_attack']:
                        attack_type = result['predicted_class']
                        attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
                    
                f.write("=== INCIDÊNCIA DE ATAQUES ===\n")
                if attack_types:
                    for attack_type, count in sorted(attack_types.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"{attack_type}: {count} ocorrências ({count/len(self.results)*100:.2f}%)\n")
                else:
                    f.write("Nenhum ataque detectado\n")
                f.write(f"Tráfego Normal: {len(benign)} ocorrências ({len(benign)/len(self.results)*100:.2f}%)\n\n")
                
                # Métricas de performance
                inference_times = [r['inference_time_ms'] for r in self.results]
                
                f.write("=== MÉTRICAS DE DESEMPENHO ===\n")
                f.write(f"Tempo médio de inferência: {np.mean(inference_times):.2f} ms\n")
                f.write(f"Tempo máximo de inferência: {np.max(inference_times):.2f} ms\n")
                f.write(f"Tempo mínimo de inferência: {np.min(inference_times):.2f} ms\n")
                f.write(f"Desvio padrão da inferência: {np.std(inference_times):.2f} ms\n")
                f.write(f"Percentil 95 (P95) da inferência: {np.percentile(inference_times, 95):.2f} ms\n")
                f.write(f"Percentil 99 (P99) da inferência: {np.percentile(inference_times, 99):.2f} ms\n")
                f.write(f"Throughput: {1000 / np.mean(inference_times):.2f} predições/segundo\n")
                
                # Adicionar métricas de CPU e memória
                stats = self.detector.get_statistics()
                f.write(f"Uso médio de CPU: {stats.get('avg_cpu_usage', 0):.2f}%\n")
                f.write(f"Uso máximo de CPU: {stats.get('max_cpu_usage', 0):.2f}%\n")
                f.write(f"Uso médio de memória: {stats.get('avg_memory_usage', 0):.2f}%\n")
                f.write(f"Uso máximo de memória: {stats.get('max_memory_usage', 0):.2f}%\n\n")
                
                # Adicionar métricas de acurácia se tivermos rótulos reais
                if self.true_labels and len(self.true_labels) == len(self.results):
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                    
                    predicted_labels = []
                    for result in self.results:
                        predicted_labels.append(1 if result['is_attack'] else 0)
                    
                    f.write("=== MÉTRICAS DE ACURÁCIA ===\n")
                    accuracy = accuracy_score(self.true_labels, predicted_labels)
                    precision = precision_score(self.true_labels, predicted_labels, zero_division=0)
                    recall = recall_score(self.true_labels, predicted_labels, zero_division=0)
                    f1 = f1_score(self.true_labels, predicted_labels, zero_division=0)
                    
                    f.write(f"Acurácia: {accuracy:.4f}\n")
                    f.write(f"Precisão: {precision:.4f}\n")
                    f.write(f"Recall: {recall:.4f}\n")
                    f.write(f"F1-Score: {f1:.4f}\n")
                    
                    # Matriz de confusão
                    cm = confusion_matrix(self.true_labels, predicted_labels)
                    f.write("\nMatriz de Confusão:\n")
                    f.write("    | Normal | Ataque\n")
                    f.write("----|--------|-------\n")
                    f.write(f"Normal  | {cm[0][0]:6d} | {cm[0][1]:6d}\n")
                    f.write(f"Ataque  | {cm[1][0]:6d} | {cm[1][1]:6d}\n\n")
                    
                    # Calcular acurácia por amostra
                    correct_predictions = sum(1 for true, pred in zip(self.true_labels, predicted_labels) if true == pred)
                    incorrect_predictions = sum(1 for true, pred in zip(self.true_labels, predicted_labels) if true != pred)
                    
                    f.write(f"Predições corretas: {correct_predictions} ({correct_predictions/len(self.results)*100:.2f}%)\n")
                    f.write(f"Predições incorretas: {incorrect_predictions} ({incorrect_predictions/len(self.results)*100:.2f}%)\n\n")
                
                    # Detalhes de todas as detecções (ataques e tráfego normal)
                f.write("=== DETALHES DE TODAS AS DETECÇÕES ===\n")
                
                # Ordenar todas as detecções por confiança
                all_results_sorted = sorted(self.results, key=lambda x: x['confidence'], reverse=True)
                
                if all_results_sorted:
                    for i, result in enumerate(all_results_sorted, 1):
                        detection_type = "ATAQUE" if result['is_attack'] else "TRÁFEGO NORMAL"
                        class_name = result['predicted_class']
                        
                        f.write(f"\n[{i}] {detection_type}: {class_name}\n")
                        f.write(f"    Timestamp: {result['timestamp']}\n")
                        f.write(f"    Confiança: {result['confidence']:.4f}\n")
                        f.write(f"    Tempo de inferência: {result['inference_time_ms']:.2f} ms\n")
                        
                        # Se temos rótulos verdadeiros, mostrar se a predição foi correta
                        if self.true_labels and i-1 < len(self.true_labels):
                            true_value = self.true_labels[i-1]  # 1 = ataque, 0 = normal
                            pred_value = 1 if result['is_attack'] else 0
                            is_correct = (true_value == pred_value)
                            f.write(f"    Predição correta: {'✓ SIM' if is_correct else '✗ NÃO'}\n")
                            
                        # Mostrar as top 3 classes com maior probabilidade
                        top_classes = sorted(
                            zip(self.detector.classes, result['all_probabilities']), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:3]
                        
                        f.write("    Top 3 classes mais prováveis:\n")
                        for cls_name, prob in top_classes:
                            f.write(f"      - {cls_name}: {prob:.4f}\n")
                        
                        f.write("    " + "-"*40 + "\n")
                        
                        # Limitar o número de detalhes exibidos para não tornar o arquivo muito grande
                        if i >= 1000:  # Limitar a 1000 resultados detalhados
                            f.write(f"\n... mais {len(all_results_sorted) - 1000} detecções omitidas ...\n")
                            break
                else:
                    f.write("Nenhuma detecção registrada durante a análise\n\n")
                    
                # Manter seções específicas para métricas gerais
                f.write("\n=== RESUMO DE ATAQUES ===\n")
                if attacks:
                    f.write(f"Total de ataques: {len(attacks)}\n")
                    attack_confidence = [r['confidence'] for r in attacks]
                    f.write(f"Confiança média de ataques: {np.mean(attack_confidence):.4f}\n")
                    f.write(f"Confiança mínima de ataques: {np.min(attack_confidence):.4f}\n")
                    f.write(f"Confiança máxima de ataques: {np.max(attack_confidence):.4f}\n")
                else:
                    f.write("Nenhum ataque detectado\n")
                
                f.write("\n=== RESUMO DE TRÁFEGO NORMAL ===\n")
                if benign:
                    f.write(f"Total de tráfego normal: {len(benign)}\n")
                    benign_confidence = [r['confidence'] for r in benign]
                    f.write(f"Confiança média de tráfego normal: {np.mean(benign_confidence):.4f}\n")
                    f.write(f"Confiança mínima de tráfego normal: {np.min(benign_confidence):.4f}\n")
                    f.write(f"Confiança máxima de tráfego normal: {np.max(benign_confidence):.4f}\n")
                else:
                    f.write("Nenhum tráfego normal detectado\n")
                    
                # Resumo das Top 10 detecções com maior confiança
                f.write("\n=== TOP 10 DETECÇÕES POR CONFIANÇA ===\n")
                top_confidence = sorted(self.results, key=lambda x: x['confidence'], reverse=True)[:10]
                
                for i, result in enumerate(top_confidence, 1):
                    detection_type = "ATAQUE" if result['is_attack'] else "NORMAL"
                    f.write(f"{i}. [{detection_type}] {result['predicted_class']} (Confiança: {result['confidence']:.4f})\n")
                
                # Resumo das 10 detecções com inferência mais rápida/lenta
                f.write("\n=== 10 INFERÊNCIAS MAIS RÁPIDAS ===\n")
                fastest = sorted(self.results, key=lambda x: x['inference_time_ms'])[:10]
                
                for i, result in enumerate(fastest, 1):
                    detection_type = "ATAQUE" if result['is_attack'] else "NORMAL" 
                    f.write(f"{i}. [{detection_type}] {result['inference_time_ms']:.2f} ms - {result['predicted_class']}\n")
                
                f.write("\n=== 10 INFERÊNCIAS MAIS LENTAS ===\n")
                slowest = sorted(self.results, key=lambda x: x['inference_time_ms'], reverse=True)[:10]
                
                for i, result in enumerate(slowest, 1):
                    detection_type = "ATAQUE" if result['is_attack'] else "NORMAL"
                    f.write(f"{i}. [{detection_type}] {result['inference_time_ms']:.2f} ms - {result['predicted_class']}\n")
    
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
