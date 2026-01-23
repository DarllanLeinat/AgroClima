import pandas as pd
import sys
import os


def calculate_weather_features(df):
    """
    Calcula features de clima (rolling averages/sums)
    a partir de um DataFrame de dados diários.
    """
    print("Calculando features de clima (médias móveis de 7 dias)...")

    if 'temp_max' in df.columns and 'temp_min' in df.columns:
        df['temp_media'] = (df['temp_max'] + df['temp_min']) / 2
        df['temp_media_movel_7d'] = df['temp_media'].rolling(window=7, min_periods=1).mean()
    else:
        print("Aviso: Colunas 'temp_max' e 'temp_min' não encontradas.", file=sys.stderr)

    if 'precipitacao' in df.columns:
        df['precip_acumulada_7d'] = df['precipitacao'].rolling(window=7, min_periods=1).sum()
    else:
        print("Aviso: Coluna 'precipitacao' não encontrada.", file=sys.stderr)

    print("Cálculos de clima concluídos.")
    return df


def load_weather_from_csv(filepath, region_filter=None):  # Removido filtro
    """
    Carrega dados meteorológicos e calcula features.
    ADAPTADO (separador ';') e CORRIGIDO (espaços nas colunas).
    """
    print(f"Carregando dados meteorológicos de '{filepath}'...")
    if not os.path.exists(filepath):
        print(f"Erro: Arquivo CSV de Clima não encontrado em '{filepath}'", file=sys.stderr)
        return pd.DataFrame()

    try:
        # --- CORREÇÃO AQUI ---
        # 1. Ler o CSV com o separador correto: ponto e vírgula (;)
        df = pd.read_csv(
            filepath,
            sep=';',  # <-- CORRIGIDO
            decimal='.'
        )

        # 2. Limpar TODOS os nomes de colunas (remove espaços em branco)
        df.columns = df.columns.str.strip()
        # --- FIM DA CORREÇÃO ---

        # 3. Verificar se a coluna 'Data' existe
        if 'Data' not in df.columns:
            print("Erro: A coluna 'Data' não foi encontrada no arquivo weather_data_mt.csv.", file=sys.stderr)
            print(f"Colunas encontradas: {df.columns.tolist()}", file=sys.stderr)
            return pd.DataFrame()

        # 4. Converter a data manualmente
        df['date'] = pd.to_datetime(df['Data'])

        # 5. Renomear colunas
        df = df.rename(columns={
            'Precipitacao (mm)': 'precipitacao',
            'Temp_Max (°C)': 'temp_max',
            'Temp_Min (°C)': 'temp_min'
        })

        # 6. Definir o 'date' como índice
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)

        # 7. Selecionar colunas relevantes e ordenar
        cols_relevantes = ['precipitacao', 'temp_max', 'temp_min']

        cols_presentes = [col for col in cols_relevantes if col in df.columns]
        if not cols_presentes:
            print("Erro: Nenhuma coluna de clima (precipitacao, temp_max, temp_min) encontrada após renomear.",
                  file=sys.stderr)
            return pd.DataFrame()

        df_final = df[cols_presentes].sort_index()

        # 8. Chamar a função de cálculo de features
        df_com_features = calculate_weather_features(df_final)

        print("Dados de clima carregados e processados com sucesso.")
        return df_com_features

    except Exception as e:
        print(f"Erro ao carregar ou processar arquivo CSV de Clima: {e}", file=sys.stderr)
        return pd.DataFrame()


if __name__ == '__main__':
    # Teste rápido
    test_path = os.path.join('data', 'weather_data_mt.csv')
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Pasta 'data' criada. Adicione 'weather_data_mt.csv' lá para testar.")
    else:
        print("Testando o carregador de clima...")
        # Não passa mais o filtro de região
        weather_data = load_weather_from_csv(test_path)
        if not weather_data.empty:
            print("\n--- Teste weather_fetcher.py OK ---")
            print(weather_data.head())
            print("-----------------------------------")
        else:
            print("\n--- Teste weather_fetcher.py FALHOU ---")