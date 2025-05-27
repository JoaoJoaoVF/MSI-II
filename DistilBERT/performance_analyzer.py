#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import onnxruntime as ort
import numpy as np
import pandas as pd
import pickle
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import argparse
import sys
from datetime import datetime

class PerformanceAnalyzer:
    def __init__(self, model_path, metadata_path):
        
        print("Carregando modelo para análise...")
        self.session = ort.InferenceSession(model_path)
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.scaler = self.metadata['scaler']
        self.label_encoder = self.metadata['label_encoder']
        self.feature_names = self.metadata['feature_names']
        self.classes = self.metadata['classes']
        
        print(f"Modelo carregado: {len(self.classes)} classes")
    
    def preprocess_data(self, df):
        
        X = df[self.feature_names].fillna(0)
        y_true_labels = df['label']
        
        X_scaled = self.scaler.transform(X)
        
        y_true = self.label_encoder.transform(y_true_labels)
        
        return X_scaled.astype(np.float32), y_true, y_true_labels
    
    def batch_predict(self, X):
        
        predictions = []
        probabilities = []
        inference_times = []
        
        batch_size = 32
        num_batches = len(X) // batch_size + (1 if len(X) % batch_size != 0 else 0)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))
            batch = X[start_idx:end_idx]
            
            start_time = time.time()
            ort_inputs = {'features': batch}
            logits, probs = self.session.run(None, ort_inputs)
            inference_time = (time.time() - start_time) * 1000
            
            batch_predictions = np.argmax(probs, axis=1)
            predictions.extend(batch_predictions)
            probabilities.extend(probs)
            inference_times.append(inference_time)
            
            if (i + 1) % 10 == 0:
                print(f"Processado batch {i+1}/{num_batches}")
        
        return np.array(predictions), np.array(probabilities), inference_times
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            classification_report, confusion_matrix
        )
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.classes,
            output_dict=True
        )
        
        cm = confusion_matrix(y_true, y_pred)
        
        benign_class_idx = 0
        if 'Benign' in self.classes:
            benign_class_idx = list(self.classes).index('Benign')
        
        y_binary_true = (y_true != benign_class_idx).astype(int)  # 1 = ataque, 0 = normal
        y_binary_pred = (y_pred != benign_class_idx).astype(int)
        
        attack_precision = precision_score(y_binary_true, y_binary_pred)
        attack_recall = recall_score(y_binary_true, y_binary_pred)
        attack_f1 = f1_score(y_binary_true, y_binary_pred)
        
        return {
            'overall': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'attack_detection': {
                'precision': attack_precision,
                'recall': attack_recall,
                'f1_score': attack_f1
            },
            'per_class': class_report,
            'confusion_matrix': cm
        }
    
    def analyze_performance(self, test_file, output_dir='analysis_results'):
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Carregando dados de teste: {test_file}")
        df = pd.read_csv(test_file)
        
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
            print(f"Usando amostra de {len(df)} registros")
        
        X, y_true, y_true_labels = self.preprocess_data(df)
        
        print("Executando predições...")
        y_pred, y_prob, inference_times = self.batch_predict(X)
        
        print("Calculando métricas...")
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        total_inference_time = sum(inference_times)
        avg_inference_time = np.mean(inference_times)
        throughput = len(X) / (total_inference_time / 1000)  # amostras por segundo
        
        performance_metrics = {
            'total_samples': len(X),
            'total_time_ms': total_inference_time,
            'avg_time_per_batch_ms': avg_inference_time,
            'avg_time_per_sample_ms': total_inference_time / len(X),
            'throughput_samples_per_second': throughput
        }
        
        self.generate_reports(metrics, performance_metrics, y_true, y_pred, y_prob, output_dir)
        
        return metrics, performance_metrics
    
    def generate_reports(self, metrics, performance_metrics, y_true, y_pred, y_prob, output_dir):
        
        report_text = f"""
# Relatório de Performance - Detecção de Ataques de Rede
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Métricas Gerais
- Accuracy: {metrics['overall']['accuracy']:.4f}
- Precision: {metrics['overall']['precision']:.4f}
- Recall: {metrics['overall']['recall']:.4f}
- F1-Score: {metrics['overall']['f1_score']:.4f}

## Detecção de Ataques (Binário)
- Precision: {metrics['attack_detection']['precision']:.4f}
- Recall: {metrics['attack_detection']['recall']:.4f}
- F1-Score: {metrics['attack_detection']['f1_score']:.4f}

## Performance de Inferência
- Total de amostras: {performance_metrics['total_samples']:,}
- Tempo total: {performance_metrics['total_time_ms']:.2f} ms
- Tempo por amostra: {performance_metrics['avg_time_per_sample_ms']:.2f} ms
- Throughput: {performance_metrics['throughput_samples_per_second']:.2f} amostras/segundo

## Métricas por Classe
"""
        
        for class_name in self.classes:
            if class_name in metrics['per_class']:
                class_metrics = metrics['per_class'][class_name]
                report_text += f"""
### {class_name}
- Precision: {class_metrics['precision']:.4f}
- Recall: {class_metrics['recall']:.4f}
- F1-Score: {class_metrics['f1-score']:.4f}
- Support: {class_metrics['support']}
"""
        
        with open(f"{output_dir}/performance_report.md", 'w') as f:
            f.write(report_text)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            metrics['confusion_matrix'], 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes
        )
        plt.title('Matriz de Confusão')
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 6))
        
        max_probs = np.max(y_prob, axis=1)
        correct_predictions = (y_true == y_pred)
        
        plt.subplot(1, 2, 1)
        plt.hist(max_probs[correct_predictions], bins=50, alpha=0.7, label='Corretas', color='green')
        plt.hist(max_probs[~correct_predictions], bins=50, alpha=0.7, label='Incorretas', color='red')
        plt.xlabel('Confiança Máxima')
        plt.ylabel('Frequência')
        plt.title('Distribuição de Confiança')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        class_counts = pd.Series(y_true).value_counts().sort_index()
        class_names = [self.classes[i] for i in class_counts.index]
        plt.bar(range(len(class_names)), class_counts.values)
        plt.xlabel('Classes')
        plt.ylabel('Número de Amostras')
        plt.title('Distribuição de Classes')
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confidence_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        results = {
            'metrics': metrics,
            'performance': performance_metrics,
            'model_info': {
                'classes': self.classes.tolist(),
                'num_features': len(self.feature_names),
                'feature_names': self.feature_names
            },
            'timestamp': datetime.now().isoformat()
        }
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        def clean_for_json(data):
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(v) for v in data]
            else:
                return convert_numpy(data)
        
        results_clean = clean_for_json(results)
        
        with open(f"{output_dir}/metrics.json", 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        print(f"\nRelatórios salvos em: {output_dir}/")
        print("- performance_report.md")
        print("- confusion_matrix.png")
        print("- confidence_distribution.png")
        print("- metrics.json")

def main():
    parser = argparse.ArgumentParser(description='Analisador de Performance para Detecção de Ataques')
    parser.add_argument('--model', default='network_attack_detector_quantized.onnx', help='Modelo ONNX')
    parser.add_argument('--metadata', default='model_metadata.pkl', help='Metadados do modelo')
    parser.add_argument('--test_data', required=True, help='Arquivo CSV de teste')
    parser.add_argument('--output', default='analysis_results', help='Diretório de saída')
    
    args = parser.parse_args()
    
    try:
        analyzer = PerformanceAnalyzer(args.model, args.metadata)
        metrics, performance = analyzer.analyze_performance(args.test_data, args.output)
        
        print("\n=== RESUMO DA ANÁLISE ===")
        print(f"Accuracy: {metrics['overall']['accuracy']:.3f}")
        print(f"Detecção de ataques F1: {metrics['attack_detection']['f1_score']:.3f}")
        print(f"Throughput: {performance['throughput_samples_per_second']:.1f} amostras/segundo")
        print(f"Tempo por amostra: {performance['avg_time_per_sample_ms']:.2f} ms")
        
    except Exception as e:
        print(f"Erro na análise: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 