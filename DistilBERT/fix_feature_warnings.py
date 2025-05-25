#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para Corrigir Avisos de Feature Names
Demonstra como resolver problemas com StandardScaler e nomes de features
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler

def demonstrate_warning_issue():
    """Demonstrar o problema dos avisos de feature names"""
    
    print("🔍 Demonstrando o problema dos avisos de feature names...")
    
    # Criar dados de exemplo com nomes de features
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    
    # Dados de treinamento (DataFrame com nomes)
    train_data = pd.DataFrame({
        'feature_1': [1, 2, 3, 4, 5],
        'feature_2': [2, 4, 6, 8, 10],
        'feature_3': [0.5, 1.0, 1.5, 2.0, 2.5],
        'feature_4': [10, 20, 30, 40, 50]
    })
    
    print(f"Dados de treinamento (DataFrame):")
    print(train_data.head())
    print(f"Tipo: {type(train_data)}")
    print()
    
    # Treinar scaler
    scaler = StandardScaler()
    scaler.fit(train_data)
    print("✅ Scaler treinado com DataFrame (com nomes de features)")
    print()
    
    # Problema: usar array NumPy para transformação
    print("❌ PROBLEMA: Usando array NumPy (sem nomes de features)")
    test_array = np.array([[1.5, 3.0, 0.75, 15.0]])
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result_array = scaler.transform(test_array)
        if w:
            print(f"⚠️  Aviso capturado: {w[0].message}")
    print()
    
    # Solução: usar DataFrame para transformação
    print("✅ SOLUÇÃO: Usando DataFrame (com nomes de features)")
    test_df = pd.DataFrame([[1.5, 3.0, 0.75, 15.0]], columns=feature_names)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result_df = scaler.transform(test_df)
        if w:
            print(f"⚠️  Aviso capturado: {w[0].message}")
        else:
            print("✅ Nenhum aviso gerado!")
    
    print(f"Resultado (array): {result_array}")
    print(f"Resultado (DataFrame): {result_df}")
    print(f"Resultados são iguais: {np.allclose(result_array, result_df)}")

def demonstrate_fix_in_monitor():
    """Demonstrar como a correção funciona no monitor"""
    
    print("\n" + "="*60)
    print("🔧 Demonstrando a correção no NetworkAttackDetector...")
    
    # Simular metadados do modelo
    feature_names = [
        'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
        'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number'
    ]
    
    # Criar scaler de exemplo
    scaler = StandardScaler()
    dummy_data = pd.DataFrame(np.random.randn(100, len(feature_names)), 
                             columns=feature_names)
    scaler.fit(dummy_data)
    
    print(f"Features esperadas: {feature_names}")
    print()
    
    # Simular dados de entrada (como dicionário)
    features_dict = {
        'flow_duration': 1500.0,
        'Header_Length': 64,
        'Protocol Type': 6,
        'Duration': 800.0,
        'Rate': 1200.0,
        'Srate': 150.0,
        'Drate': 120.0,
        'fin_flag_number': 1,
        'syn_flag_number': 0
    }
    
    print("Dados de entrada (dicionário):")
    for key, value in features_dict.items():
        print(f"  {key}: {value}")
    print()
    
    # Método ANTIGO (problemático)
    print("❌ Método antigo (gera avisos):")
    features_array = np.array([features_dict.get(name, 0.0) for name in feature_names])
    print(f"Array criado: {features_array}")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result_old = scaler.transform(features_array.reshape(1, -1))
        if w:
            print(f"⚠️  Aviso: {w[0].message}")
    print()
    
    # Método NOVO (corrigido)
    print("✅ Método novo (sem avisos):")
    feature_values = {}
    for feature_name in feature_names:
        feature_values[feature_name] = features_dict.get(feature_name, 0.0)
    
    features_df = pd.DataFrame([feature_values])
    print(f"DataFrame criado: {features_df.shape}")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result_new = scaler.transform(features_df)
        if w:
            print(f"⚠️  Aviso: {w[0].message}")
        else:
            print("✅ Nenhum aviso gerado!")
    
    print(f"Resultados são iguais: {np.allclose(result_old, result_new)}")

def show_suppression_methods():
    """Mostrar métodos para suprimir avisos"""
    
    print("\n" + "="*60)
    print("🔇 Métodos para suprimir avisos...")
    
    methods = [
        {
            'name': 'Filtro específico por mensagem',
            'code': "warnings.filterwarnings('ignore', message='X does not have valid feature names')"
        },
        {
            'name': 'Filtro por categoria e módulo',
            'code': "warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')"
        },
        {
            'name': 'Contexto temporário',
            'code': """
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = scaler.transform(data)
"""
        },
        {
            'name': 'Variável de ambiente',
            'code': "export PYTHONWARNINGS='ignore::UserWarning'"
        }
    ]
    
    for i, method in enumerate(methods, 1):
        print(f"{i}. {method['name']}:")
        print(f"   {method['code']}")
        print()

def main():
    """Função principal"""
    print("🛠️  Correção de Avisos de Feature Names - StandardScaler")
    print("="*60)
    
    # Demonstrar o problema
    demonstrate_warning_issue()
    
    # Demonstrar a correção no monitor
    demonstrate_fix_in_monitor()
    
    # Mostrar métodos de supressão
    show_suppression_methods()
    
    print("="*60)
    print("📋 RESUMO DAS CORREÇÕES APLICADAS:")
    print("1. ✅ Usar DataFrame em vez de array NumPy")
    print("2. ✅ Manter nomes das features durante transformação")
    print("3. ✅ Adicionar filtros de avisos específicos")
    print("4. ✅ Tratamento de erro com fallback")
    print()
    print("🎯 O sistema agora deve funcionar sem avisos!")

if __name__ == '__main__':
    main() 