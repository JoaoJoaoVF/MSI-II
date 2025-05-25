#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste da Funcionalidade de Arquivo de Resultado
Demonstra como o sistema gera arquivos de resultado baseados no nome do CSV
"""

import os
import pandas as pd
import numpy as np

def create_sample_csv(filename, num_samples=100):
    """Criar um arquivo CSV de exemplo para teste"""
    
    # Criar dados de exemplo
    np.random.seed(42)  # Para resultados reproduzíveis
    
    # Features de exemplo (simulando dados de rede)
    data = {
        'flow_duration': np.random.exponential(1000, num_samples),
        'Header_Length': np.random.randint(20, 1500, num_samples),
        'Protocol Type': np.random.choice([6, 17, 1], num_samples),  # TCP, UDP, ICMP
        'Duration': np.random.exponential(500, num_samples),
        'Rate': np.random.exponential(1000, num_samples),
        'Srate': np.random.exponential(100, num_samples),
        'Drate': np.random.exponential(100, num_samples),
        'fin_flag_number': np.random.randint(0, 2, num_samples),
        'syn_flag_number': np.random.randint(0, 2, num_samples),
        'rst_flag_number': np.random.randint(0, 2, num_samples),
        'psh_flag_number': np.random.randint(0, 2, num_samples),
        'ack_flag_number': np.random.randint(0, 2, num_samples),
        'ece_flag_number': np.random.randint(0, 2, num_samples),
        'cwr_flag_number': np.random.randint(0, 2, num_samples),
        'HTTP': np.random.randint(0, 2, num_samples),
        'HTTPS': np.random.randint(0, 2, num_samples),
        'DNS': np.random.randint(0, 2, num_samples),
        'Telnet': np.random.randint(0, 2, num_samples),
        'SMTP': np.random.randint(0, 2, num_samples),
        'SSH': np.random.randint(0, 2, num_samples),
        'IRC': np.random.randint(0, 2, num_samples),
        'TCP': np.random.randint(0, 2, num_samples),
        'UDP': np.random.randint(0, 2, num_samples),
        'DHCP': np.random.randint(0, 2, num_samples),
        'ARP': np.random.randint(0, 2, num_samples),
        'ICMP': np.random.randint(0, 2, num_samples),
        'IPv': np.random.randint(0, 2, num_samples),
        'LLC': np.random.randint(0, 2, num_samples),
        'Tot sum': np.random.randint(0, 10000, num_samples),
        'Min': np.random.randint(0, 1000, num_samples),
        'Max': np.random.randint(1000, 10000, num_samples),
        'AVG': np.random.randint(100, 5000, num_samples),
        'Std': np.random.randint(0, 1000, num_samples),
        'Tot size': np.random.randint(0, 100000, num_samples),
        'IAT': np.random.exponential(100, num_samples),
        'Number': np.random.randint(1, 1000, num_samples),
        'Magnitue': np.random.exponential(1000, num_samples),
        'Radius': np.random.exponential(500, num_samples),
        'Covariance': np.random.normal(0, 1, num_samples),
        'Variance': np.random.exponential(100, num_samples),
        'Weight': np.random.exponential(10, num_samples),
    }
    
    # Labels de exemplo
    labels = np.random.choice([
        'Benign', 'DDoS-SYN_Flood', 'DDoS-ICMP_Flood', 'DDoS-UDP_Flood',
        'DoS-HTTP_Flood', 'Recon-PortScan', 'SqlInjection', 'XSS'
    ], num_samples, p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.03, 0.03, 0.04])
    
    data['label'] = labels
    
    # Criar DataFrame e salvar
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
    print(f"✅ Arquivo CSV de teste criado: {filename}")
    print(f"   - {num_samples} amostras")
    print(f"   - {len(data)-1} features")
    print(f"   - Labels: {np.unique(labels)}")
    
    return filename

def test_result_filename():
    """Testar a geração de nomes de arquivos de resultado"""
    
    test_cases = [
        "network_data.csv",
        "attack_samples.csv", 
        "test_data_2024.csv",
        "dados_rede.csv"
    ]
    
    print("🔍 Testando geração de nomes de arquivos de resultado:")
    print("=" * 60)
    
    for csv_file in test_cases:
        # Simular a lógica do sistema
        csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
        result_file = f"result-{csv_basename}.txt"
        
        print(f"CSV: {csv_file}")
        print(f"  → Resultado: {result_file}")
        print()

def main():
    """Função principal de teste"""
    print("🧪 Teste da Funcionalidade de Arquivo de Resultado")
    print("=" * 60)
    
    # Testar geração de nomes
    test_result_filename()
    
    # Criar arquivo CSV de exemplo
    sample_file = "sample_network_data.csv"
    create_sample_csv(sample_file, 50)
    
    print("\n" + "=" * 60)
    print("📋 Como usar com o sistema real:")
    print(f"python3 realtime_network_monitor.py --simulate {sample_file}")
    print(f"  → Criará: result-sample_network_data.txt")
    print()
    print("Ou com arquivo personalizado:")
    print(f"python3 realtime_network_monitor.py --simulate {sample_file} --output meu_resultado.txt")
    print("  → Criará: meu_resultado.txt")
    
    print("\n✅ Teste concluído!")
    print(f"📁 Arquivo de exemplo criado: {sample_file}")

if __name__ == '__main__':
    main() 