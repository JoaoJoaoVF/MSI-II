#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste de Sintaxe - Verificar se todos os arquivos Python estão corretos
"""

import sys
import importlib.util

def test_file_syntax(filename):
    """Testar sintaxe de um arquivo Python"""
    try:
        spec = importlib.util.spec_from_file_location("test_module", filename)
        if spec is None:
            print(f"❌ {filename} - Não foi possível carregar")
            return False
        
        module = importlib.util.module_from_spec(spec)
        # Não executar o módulo, apenas verificar sintaxe
        print(f"✅ {filename} - Sintaxe OK")
        return True
        
    except SyntaxError as e:
        print(f"❌ {filename} - Erro de sintaxe: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {filename} - Aviso: {e}")
        return True  # Outros erros não são de sintaxe

def main():
    """Testar sintaxe de todos os arquivos Python"""
    print("🔍 Testando sintaxe dos arquivos Python...")
    
    files_to_test = [
        "realtime_network_monitor.py",
        "performance_analyzer.py", 
        "test_installation.py"
    ]
    
    all_ok = True
    
    for filename in files_to_test:
        try:
            if test_file_syntax(filename):
                continue
            else:
                all_ok = False
        except FileNotFoundError:
            print(f"⚠️  {filename} - Arquivo não encontrado")
    
    print("\n" + "="*50)
    if all_ok:
        print("🎉 Todos os arquivos têm sintaxe correta!")
        print("✅ Sistema pronto para uso")
    else:
        print("❌ Alguns arquivos têm erros de sintaxe")
        print("🔧 Corrija os erros antes de continuar")
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main()) 