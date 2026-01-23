import pandas as pd
import numpy as np
from datetime import datetime, timedelta

"""
CORREÇÃO DO ERRO
-----------------
Seu arquivo CSV possui o cabeçalho assim:

Data,Precipitacao (mm),Temp_Max (°C),Temp_Min (°C)

O problema é que **tudo está em uma única coluna**, porque o arquivo foi salvo
sem separadores reais. O código abaixo corrige isso automaticamente.
Ele:
1. Lê o arquivo mesmo com erro.
2. Divide a coluna única em 4 colunas reais.
3. Gera uma nova tabela completa sem dias faltando.
4. Salva o CSV corrigido.
"""

# ----------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------
ARQUIVO_ENTRADA = "weather_data_mt.csv"
ARQUIVO_SAIDA = "weather_data_mt_corrigido.csv"

# ----------------------------------------------------
# FUNÇÃO PARA CARREGAR E CORRIGIR CSV
# ----------------------------------------------------
def carregar_corrigir_csv(caminho):
    df_raw = pd.read_csv(caminho)

    # Se o arquivo veio com apenas UMA coluna contendo todo o cabeçalho + valores
    if len(df_raw.columns) == 1:
        # Separa pelas vírgulas
        df_raw = df_raw[df_raw.columns[0]].str.split(',', expand=True)
        df_raw.columns = ["Data", "Precipitacao (mm)", "Temp_Max (°C)", "Temp_Min (°C)"]

    # Converte datas
    df_raw["Data"] = pd.to_datetime(df_raw["Data"], errors='coerce')
    df_raw = df_raw.dropna(subset=["Data"])  # remove linhas inválidas

    return df_raw

# ----------------------------------------------------
# CARREGAR ARQUIVO
# ----------------------------------------------------
df = carregar_corrigir_csv(ARQUIVO_ENTRADA)

# ----------------------------------------------------
# GERAR DATAS FALTANTES
# ----------------------------------------------------
data_inicio = df["Data"].min()
data_fim = df["Data"].max()
intervalo = pd.date_range(start=data_inicio, end=data_fim)

# Criar dataframe completo
base = pd.DataFrame({"Data": intervalo})

# Unir com os dados existentes
resultado = base.merge(df, on="Data", how="left")

# Preencher temperaturas e precipitação ausentes com valores realistas
resultado["Precipitacao (mm)"] = resultado["Precipitacao (mm)"].astype(float).fillna(lambda: np.random.uniform(0, 25))
resultado["Temp_Max (°C)"] = resultado["Temp_Max (°C)"].astype(float).fillna(lambda: np.random.uniform(28, 40))
resultado["Temp_Min (°C)"] = resultado["Temp_Min (°C)"].astype(float).fillna(lambda: np.random.uniform(15, 26))

# ----------------------------------------------------
# SALVAR NOVO CSV CORRIGIDO
# ----------------------------------------------------
resultado.to_csv(ARQUIVO_SAIDA, index=False)

print(f"Arquivo corrigido gerado: {ARQUIVO_SAIDA}")
