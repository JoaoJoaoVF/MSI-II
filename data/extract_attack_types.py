import pandas as pd
import os
import glob

# Caminho para o diretório contendo os arquivos CSV
data_dir = r"c:\Users\jvfer\OneDrive\Treinamentos MSI II\MSI-II\data"

# Encontrar todos os arquivos CSV no diretório
csv_files = glob.glob(os.path.join(data_dir, "part-*.csv"))

# Conjunto para armazenar valores únicos da coluna 'label'
unique_attacks = set()

# Processar cada arquivo CSV
for file_path in csv_files:
    try:
        print(f"Processando: {os.path.basename(file_path)}")
        # Ler o arquivo CSV
        df = pd.read_csv(file_path)
        
        # Verificar se a coluna 'label' existe
        if 'label' in df.columns:
            # Adicionar valores únicos ao conjunto
            unique_attacks.update(df['label'].unique())
        else:
            print(f"Aviso: O arquivo {os.path.basename(file_path)} não contém a coluna 'label'")
            
    except Exception as e:
        print(f"Erro ao processar {os.path.basename(file_path)}: {str(e)}")

# Exibir a lista de ataques únicos
print("\nTipos de ataques encontrados:")
for i, attack in enumerate(sorted(unique_attacks), 1):
    print(f"{i}. {attack}")

# Salvar a lista em um arquivo de texto
output_file = os.path.join(data_dir, "unique_attack_types.txt")
with open(output_file, 'w') as f:
    for attack in sorted(unique_attacks):
        f.write(f"{attack}\n")

print(f"\nLista de ataques salvos em: {output_file}")