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
import os
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
    def __init__(self, detector, log_file='attack_log.json', result_file=None):
        self.detector = detector
        self.log_file = log_file
        self.result_file = result_file
        self.data_queue = queue.Queue()
        self.running = False
        self.results = []  # Armazenar todos os resultados
    
    def log_detection(self, result):
        """Registrar detecção em arquivo"""
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
    
    def save_result(self, message):
        """Salvar resultado em arquivo específico"""
        
        if self.result_file:
            with open(self.result_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        else:
            print(message)
    
    def save_all_results(self):
        """Salvar todos os resultados coletados"""
        
        if self.result_file and self.results:
            with open(self.result_file, 'w', encoding='utf-8') as f:
                f.write("=== RESULTADOS DA ANÁLISE ===\n")
                f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total de amostras processadas: {len(self.results)}\n\n")
                
                # Estatísticas gerais
                attacks = [r for r in self.results if r['is_attack']]
                f.write(f"Ataques detectados: {len(attacks)}\n")
                f.write(f"Taxa de ataques: {len(attacks)/len(self.results)*100:.2f}%\n")
                f.write(f"Tráfego normal: {len(self.results) - len(attacks)}\n\n")
                
                # Tipos de ataques detectados
                if attacks:
                    attack_types = {}
                    for attack in attacks:
                        attack_type = attack['predicted_class']
                        attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
                    
                    f.write("=== TIPOS DE ATAQUES DETECTADOS ===\n")
                    for attack_type, count in sorted(attack_types.items()):
                        f.write(f"{attack_type}: {count} ocorrências\n")
                    f.write("\n")
                
                # Detalhes de cada detecção
                f.write("=== DETALHES DAS DETECÇÕES ===\n")
                for i, result in enumerate(self.results, 1):
                    status = "🚨 ATAQUE" if result['is_attack'] else "✅ NORMAL"
                    f.write(f"Amostra {i}: {status}\n")
                    f.write(f"  Classe: {result['predicted_class']}\n")
                    f.write(f"  Confiança: {result['confidence']:.3f}\n")
                    f.write(f"  Tempo de inferência: {result['inference_time_ms']:.2f} ms\n")
                    f.write(f"  Timestamp: {result['timestamp']}\n")
                    f.write("\n")
    
    def process_data_stream(self):
        """Processar stream de dados em tempo real"""
        
        while self.running:
            try:
                # Obter dados da fila
                features_dict = self.data_queue.get(timeout=1)
                
                # Fazer predição
                result = self.detector.predict(features_dict)
                
                # Armazenar resultado
                self.results.append(result)
                
                # Log se for ataque
                if result['is_attack']:
                    message = f"🚨 ATAQUE DETECTADO: {result['predicted_class']} (Confiança: {result['confidence']:.3f})"
                    self.save_result(message)
                    self.log_detection(result)
                else:
                    message = f"✅ Tráfego normal (Confiança: {result['confidence']:.3f})"
                    self.save_result(message)
                
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                error_msg = f"Erro no processamento: {e}"
                self.save_result(error_msg)
    
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
    
    message = f"Carregando dados de simulação: {csv_file}"
    monitor.save_result(message)
    df = pd.read_csv(csv_file)
    
    message = f"Iniciando simulação com {len(df)} amostras..."
    monitor.save_result(message)
    
    for idx, row in df.iterrows():
        # Converter linha para dicionário (excluindo label)
        features_dict = row.drop('label').to_dict()
        
        # Adicionar à fila de monitoramento
        monitor.add_data(features_dict)
        
        # Mostrar progresso
        if (idx + 1) % 100 == 0:
            stats = detector.get_statistics()
            progress_msg = f"\nProcessadas {idx + 1} amostras"
            monitor.save_result(progress_msg)
            monitor.save_result(f"Taxa de ataques: {stats.get('attack_rate', 0):.3f}")
            monitor.save_result(f"Tempo médio: {stats.get('avg_inference_time_ms', 0):.2f} ms")
        
        time.sleep(delay)

def main():
    parser = argparse.ArgumentParser(description='Detector de Ataques de Rede em Tempo Real')
    parser.add_argument('--model', default='network_attack_detector_quantized.onnx', help='Modelo ONNX')
    parser.add_argument('--metadata', default='model_metadata.pkl', help='Metadados do modelo')
    parser.add_argument('--simulate', type=str, help='Arquivo CSV para simulação')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay entre amostras (segundos)')
    parser.add_argument('--interactive', action='store_true', help='Modo interativo')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark de performance')
    parser.add_argument('--output', type=str, help='Arquivo de saída personalizado')
    
    args = parser.parse_args()
    
    # Criar nome do arquivo de resultado baseado no CSV de entrada
    result_file = None
    if args.simulate:
        csv_basename = os.path.splitext(os.path.basename(args.simulate))[0]
        result_file = f"result-{csv_basename}.txt"
        print(f"Resultados serão salvos em: {result_file}")
    elif args.output:
        result_file = args.output
        print(f"Resultados serão salvos em: {result_file}")
    
    # Inicializar detector
    try:
        detector = NetworkAttackDetector(args.model, args.metadata)
        monitor = RealTimeMonitor(detector, result_file=result_file)
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
        print(f"\nResultados do benchmark:")
        print(f"Predições: {stats['total_predictions']}")
        print(f"Tempo médio: {stats['avg_inference_time_ms']:.2f} ms")
        print(f"Throughput: {stats['throughput_per_second']:.2f} predições/segundo")
        
    elif args.simulate:
        # Simular dados de rede
        monitor_thread = monitor.start_monitoring()
        
        try:
            simulate_network_data(args.simulate, detector, monitor, args.delay)
            
            # Aguardar processamento de todas as amostras
            monitor.data_queue.join()
            
        except KeyboardInterrupt:
            monitor.save_result("Interrompido pelo usuário")
        finally:
            monitor.stop_monitoring()
            
            # Salvar relatório final
            monitor.save_all_results()
            
            # Mostrar estatísticas finais
            stats = detector.get_statistics()
            final_stats = f"\n=== ESTATÍSTICAS FINAIS ==="
            monitor.save_result(final_stats)
            for key, value in stats.items():
                monitor.save_result(f"{key}: {value}")
            
            if result_file:
                print(f"\n✅ Análise concluída! Resultados salvos em: {result_file}")
    
    elif args.interactive:
        # Modo interativo
        print("\nModo interativo ativado.")
        print("Digite valores para as features ou 'sair' para encerrar.")
        print(f"Features necessárias: {detector.feature_names[:5]}... (total: {len(detector.feature_names)})")
        
        while True:
            try:
                # Exemplo simples: usar valores aleatórios
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
