
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Detecção de Ataques de Rede em Tempo Real
Otimizado para Raspberry Pi
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

class NetworkAttackDetector:
    def __init__(self, model_path, metadata_path):
        """Inicializar detector de ataques"""
        
        print("Carregando modelo...")
        self.session = ort.InferenceSession(model_path)
        
        print("Carregando metadados...")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.scaler = self.metadata['scaler']
        self.label_encoder = self.metadata['label_encoder']
        self.feature_names = self.metadata['feature_names']
        self.classes = self.metadata['classes']
        
        print(f"Modelo carregado com sucesso!")
        print(f"Classes detectáveis: {self.classes}")
        
        # Estatísticas
        self.total_predictions = 0
        self.attack_detections = 0
        self.inference_times = []
    
    def preprocess_features(self, features_dict):
        """Pré-processar features de entrada"""
        
        # Converter para array na ordem correta
        features_array = np.array([features_dict.get(name, 0.0) for name in self.feature_names])
        
        # Normalizar
        features_scaled = self.scaler.transform(features_array.reshape(1, -1))
        
        return features_scaled.astype(np.float32)
    
    def predict(self, features_dict):
        """Fazer predição de ataque"""
        
        # Pré-processar
        features = self.preprocess_features(features_dict)
        
        # Inferência
        start_time = time.time()
        ort_inputs = {'features': features}
        logits, probabilities = self.session.run(None, ort_inputs)
        inference_time = (time.time() - start_time) * 1000
        
        # Processar resultado
        predicted_class_idx = np.argmax(probabilities[0])
        predicted_class = self.classes[predicted_class_idx]
        confidence = probabilities[0][predicted_class_idx]
        
        # Atualizar estatísticas
        self.total_predictions += 1
        self.inference_times.append(inference_time)
        
        if predicted_class != 'Benign':  # Assumindo que 'Benign' é tráfego normal
            self.attack_detections += 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'is_attack': predicted_class != 'Benign',
            'inference_time_ms': inference_time,
            'all_probabilities': probabilities[0].tolist()
        }
    
    def get_statistics(self):
        """Obter estatísticas do detector"""
        
        if not self.inference_times:
            return {}
        
        return {
            'total_predictions': self.total_predictions,
            'attack_detections': self.attack_detections,
            'attack_rate': self.attack_detections / self.total_predictions if self.total_predictions > 0 else 0,
            'avg_inference_time_ms': np.mean(self.inference_times),
            'max_inference_time_ms': np.max(self.inference_times),
            'min_inference_time_ms': np.min(self.inference_times),
            'throughput_per_second': 1000 / np.mean(self.inference_times)
        }

class RealTimeMonitor:
    def __init__(self, detector, log_file='attack_log.json'):
        self.detector = detector
        self.log_file = log_file
        self.data_queue = queue.Queue()
        self.running = False
    
    def log_detection(self, result):
        """Registrar detecção em arquivo"""
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(result) + '
')
    
    def process_data_stream(self):
        """Processar stream de dados em tempo real"""
        
        while self.running:
            try:
                # Obter dados da fila
                features_dict = self.data_queue.get(timeout=1)
                
                # Fazer predição
                result = self.detector.predict(features_dict)
                
                # Log se for ataque
                if result['is_attack']:
                    print(f"🚨 ATAQUE DETECTADO: {result['predicted_class']} (Confiança: {result['confidence']:.3f})")
                    self.log_detection(result)
                else:
                    print(f"✅ Tráfego normal (Confiança: {result['confidence']:.3f})")
                
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erro no processamento: {e}")
    
    def start_monitoring(self):
        """Iniciar monitoramento"""
        
        self.running = True
        monitor_thread = threading.Thread(target=self.process_data_stream)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print("Monitoramento iniciado...")
        return monitor_thread
    
    def stop_monitoring(self):
        """Parar monitoramento"""
        self.running = False
    
    def add_data(self, features_dict):
        """Adicionar dados para análise"""
        self.data_queue.put(features_dict)

def simulate_network_data(csv_file, detector, monitor, delay=1.0):
    """Simular dados de rede em tempo real"""
    
    print(f"Carregando dados de simulação: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"Iniciando simulação com {len(df)} amostras...")
    
    for idx, row in df.iterrows():
        # Converter linha para dicionário (excluindo label)
        features_dict = row.drop('label').to_dict()
        
        # Adicionar à fila de monitoramento
        monitor.add_data(features_dict)
        
        # Mostrar progresso
        if (idx + 1) % 100 == 0:
            stats = detector.get_statistics()
            print(f"
Processadas {idx + 1} amostras")
            print(f"Taxa de ataques: {stats.get('attack_rate', 0):.3f}")
            print(f"Tempo médio: {stats.get('avg_inference_time_ms', 0):.2f} ms")
        
        time.sleep(delay)

def main():
    parser = argparse.ArgumentParser(description='Detector de Ataques de Rede em Tempo Real')
    parser.add_argument('--model', default='network_attack_detector_quantized.onnx', help='Modelo ONNX')
    parser.add_argument('--metadata', default='model_metadata.pkl', help='Metadados do modelo')
    parser.add_argument('--simulate', type=str, help='Arquivo CSV para simulação')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay entre amostras (segundos)')
    parser.add_argument('--interactive', action='store_true', help='Modo interativo')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark de performance')
    
    args = parser.parse_args()
    
    # Inicializar detector
    try:
        detector = NetworkAttackDetector(args.model, args.metadata)
        monitor = RealTimeMonitor(detector)
    except Exception as e:
        print(f"Erro ao inicializar detector: {e}")
        sys.exit(1)
    
    if args.benchmark:
        # Benchmark de performance
        print("Executando benchmark...")
        
        # Criar dados de teste
        test_features = {name: np.random.randn() for name in detector.feature_names}
        
        # Executar múltiplas predições
        for i in range(1000):
            detector.predict(test_features)
        
        stats = detector.get_statistics()
        print(f"
Resultados do benchmark:")
        print(f"Predições: {stats['total_predictions']}")
        print(f"Tempo médio: {stats['avg_inference_time_ms']:.2f} ms")
        print(f"Throughput: {stats['throughput_per_second']:.2f} predições/segundo")
        
    elif args.simulate:
        # Simular dados de rede
        monitor_thread = monitor.start_monitoring()
        
        try:
            simulate_network_data(args.simulate, detector, monitor, args.delay)
        except KeyboardInterrupt:
            print("
Interrompido pelo usuário")
        finally:
            monitor.stop_monitoring()
            
            # Mostrar estatísticas finais
            stats = detector.get_statistics()
            print(f"
=== ESTATÍSTICAS FINAIS ===")
            for key, value in stats.items():
                print(f"{key}: {value}")
    
    elif args.interactive:
        # Modo interativo
        print("
Modo interativo ativado.")
        print("Digite valores para as features ou 'sair' para encerrar.")
        print(f"Features necessárias: {detector.feature_names[:5]}... (total: {len(detector.feature_names)})")
        
        while True:
            try:
                # Exemplo simples: usar valores aleatórios
                input("Pressione Enter para gerar predição com dados aleatórios (ou Ctrl+C para sair): ")
                
                test_features = {name: np.random.randn() for name in detector.feature_names}
                result = detector.predict(test_features)
                
                print(f"
Resultado:")
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
