#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Teste Rápido - Verificar Instalação
Testa se todas as dependências estão funcionando corretamente
"""

import sys
import time

def test_imports():
    """Testar importações críticas"""
    print("🔍 Testando importações...")
    
    tests = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("sklearn", None),
        ("onnxruntime", "ort"),
        ("matplotlib.pyplot", "plt"),
        ("seaborn", "sns")
    ]
    
    results = {}
    
    for module, alias in tests:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"✅ {module} - OK")
            results[module] = True
        except ImportError as e:
            print(f"❌ {module} - ERRO: {e}")
            results[module] = False
    
    return results

def test_onnx_basic():
    """Teste básico do ONNX Runtime"""
    print("\n🧠 Testando ONNX Runtime...")
    
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Verificar providers disponíveis
        providers = ort.get_available_providers()
        print(f"Providers disponíveis: {providers}")
        
        # Teste básico de sessão (sem modelo real)
        print("✅ ONNX Runtime funcionando")
        return True
        
    except Exception as e:
        print(f"❌ Erro no ONNX Runtime: {e}")
        return False

def test_data_processing():
    """Testar processamento básico de dados"""
    print("\n📊 Testando processamento de dados...")
    
    try:
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        
        # Criar dados de teste
        data = np.random.rand(100, 10)
        df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(10)])
        
        # Testar normalização
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        print(f"✅ Dados processados: {scaled_data.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Erro no processamento: {e}")
        return False

def test_visualization():
    """Testar capacidades de visualização"""
    print("\n📈 Testando visualização...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Backend sem GUI
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Criar gráfico simples
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title('Teste de Visualização')
        
        # Salvar (não mostrar)
        plt.savefig('test_plot.png')
        plt.close()
        
        print("✅ Visualização funcionando")
        return True
        
    except Exception as e:
        print(f"❌ Erro na visualização: {e}")
        return False

def test_system_info():
    """Mostrar informações do sistema"""
    print("\n💻 Informações do Sistema:")
    print(f"Python: {sys.version}")
    print(f"Plataforma: {sys.platform}")
    
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except:
        pass
    
    try:
        import pandas as pd
        print(f"Pandas: {pd.__version__}")
    except:
        pass
    
    try:
        import sklearn
        print(f"Scikit-learn: {sklearn.__version__}")
    except:
        pass
    
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime: {ort.__version__}")
    except:
        pass

def main():
    """Função principal de teste"""
    print("🚀 Teste de Instalação - Sistema de Detecção de Ataques")
    print("=" * 60)
    
    start_time = time.time()
    
    # Executar testes
    import_results = test_imports()
    onnx_ok = test_onnx_basic()
    data_ok = test_data_processing()
    viz_ok = test_visualization()
    
    # Mostrar informações do sistema
    test_system_info()
    
    # Resumo dos resultados
    print("\n" + "=" * 60)
    print("📋 RESUMO DOS TESTES:")
    
    critical_modules = ['numpy', 'pandas', 'sklearn', 'onnxruntime']
    critical_ok = all(import_results.get(mod, False) for mod in critical_modules)
    
    if critical_ok and onnx_ok and data_ok:
        print("🎉 SUCESSO: Todos os testes críticos passaram!")
        print("✅ Sistema pronto para uso")
        exit_code = 0
    else:
        print("⚠️  ATENÇÃO: Alguns testes falharam")
        print("❌ Verifique as dependências")
        exit_code = 1
    
    # Detalhes dos testes
    print(f"\nImportações críticas: {'✅' if critical_ok else '❌'}")
    print(f"ONNX Runtime: {'✅' if onnx_ok else '❌'}")
    print(f"Processamento de dados: {'✅' if data_ok else '❌'}")
    print(f"Visualização: {'✅' if viz_ok else '⚠️ '}")
    
    elapsed = time.time() - start_time
    print(f"\nTempo total: {elapsed:.2f} segundos")
    
    if exit_code == 0:
        print("\n🎯 Próximos passos:")
        print("1. Execute: python3 realtime_network_monitor.py --benchmark")
        print("2. Teste com dados: python3 realtime_network_monitor.py --simulate dados.csv")
        print("3. Análise: python3 performance_analyzer.py --test_data dados.csv")
    else:
        print("\n🔧 Para resolver problemas:")
        print("1. Consulte: TROUBLESHOOTING.md")
        print("2. Execute: ./install_manual.sh")
        print("3. Verifique: pip list")
    
    return exit_code

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 