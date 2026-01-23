import pandas as pd
import sys
import os
from sqlalchemy import create_engine
import warnings

# --- Configuração do Pipeline ---
DATA_DIR = "data"
IBGE_PRODUCAO_FILE = os.path.join(DATA_DIR, "ibge_producao_mt.csv")
CEPEA_PRECOS_FILE = os.path.join(DATA_DIR, "cepea_precos.csv")
WEATHER_FILE = os.path.join(DATA_DIR, "weather_data_mt.csv")
TARGET_REGION = "Mato Grosso"
DB_PATH = os.path.join(DATA_DIR, "agriforecast_unified_db.sqlite")
DB_TABLE_NAME = "unified_data"


# --- FUNÇÕES DO CARREGADOR ---

def load_ibge_from_csv(filepath, region_filter=TARGET_REGION):
    """
    Carrega os dados de produção do IBGE (ibge_producao_mt.csv).
    """
    print(f"Carregando dados do IBGE de '{filepath}'...")
    if not os.path.exists(filepath):
        print(f"Erro: Arquivo CSV do IBGE não encontrado em '{filepath}'", file=sys.stderr)
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath, sep=',', encoding='utf-8', decimal='.')
        meses_map = {
            'Janeiro': 1, 'Fevereiro': 2, 'Março': 3, 'Abril': 4, 'Maio': 5, 'Junho': 6,
            'Julho': 7, 'Agosto': 8, 'Setembro': 9, 'Outubro': 10, 'Novembro': 11, 'Dezembro': 12
        }
        df['mes_num'] = df['mes'].map(meses_map)
        df['date'] = pd.to_datetime(
            df[['ano', 'mes_num', 'dia']].rename(columns={'ano': 'year', 'mes_num': 'month', 'dia': 'day'}))
        df_filtrado = df[df['regiao'].str.strip() == region_filter]
        df_filtrado = df_filtrado.pivot(index='date', columns='produto', values='quantidade (toneladas)')
        df_filtrado.columns = [f'producao_{col.lower().replace(" (grão)", "").replace(" 1ª safra", "")}' for col in
                               df_filtrado.columns]
        df_filtrado = df_filtrado.apply(pd.to_numeric, errors='coerce')
        print(f"Dados do IBGE carregados e filtrados para '{region_filter}'.")
        return df_filtrado
    except Exception as e:
        print(f"Erro ao carregar ou processar arquivo IBGE: {e}", file=sys.stderr)
        return pd.DataFrame()


def load_cepea_from_csv(filepath):
    """
    Carrega dados de preços do CEPEA (cepea_precos.csv).
    """
    print(f"Carregando dados de preços do CEPEA de {filepath}...")
    if not os.path.exists(filepath):
        print(f"Erro: Arquivo CSV do CEPEA não encontrado em '{filepath}'", file=sys.stderr)
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath, sep=',', parse_dates=['Data'])
        df = df.rename(columns={'Data': 'date'})
        df.set_index('date', inplace=True)
        df.columns = [f'preco_{col.lower().split(" (")[0].replace(" (", "")}' for col in df.columns]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        print("Dados do CEPEA carregados com sucesso.")
        return df
    except Exception as e:
        print(f"Erro ao carregar ou processar arquivo CEPEA: {e}", file=sys.stderr)
        return pd.DataFrame()


# --- FUNÇÃO DE CLIMA (CORRIGIDA PARA SEPARADOR ';') ---
def load_weather_data(filepath, region_filter=None):  # Removido o filtro, não é mais usado aqui
    """
    Carrega dados meteorológicos do CSV original (weather_data_mt.csv)
    que usa ponto e vírgula (;) e não tem coluna 'Estado'.
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
            decimal='.'  # Decimal provavelmente é '.' mesmo
        )

        # 2. Limpar TODOS os nomes de colunas (remove espaços em branco)
        df.columns = df.columns.str.strip()

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
        # --- FIM DA CORREÇÃO ---

        # 6. Definir o 'date' como índice
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)

        # 7. Selecionar colunas relevantes e ordenar
        cols_relevantes = ['precipitacao', 'temp_max', 'temp_min']

        cols_presentes = [col for col in cols_relevantes if col in df.columns]
        if len(cols_presentes) != len(cols_relevantes):
            print(f"Aviso: Nem todas as colunas de clima foram encontradas. Colunas presentes: {cols_presentes}",
                  file=sys.stderr)
            if not cols_presentes:
                print("Erro: Nenhuma coluna de clima encontrada após renomear.", file=sys.stderr)
                return pd.DataFrame()

        df_final = df[cols_presentes].sort_index()

        # 8. Calcular features
        if 'temp_max' in df_final.columns and 'temp_min' in df_final.columns:
            df_final['temp_media'] = (df_final['temp_max'] + df_final['temp_min']) / 2
            df_final['temp_media_movel_7d'] = df_final['temp_media'].rolling(window=7, min_periods=1).mean()

        if 'precipitacao' in df_final.columns:
            df_final['precip_acumulada_7d'] = df_final['precipitacao'].rolling(window=7, min_periods=1).sum()

        print("Dados de clima carregados e processados com sucesso.")
        return df_final
    except Exception as e:
        print(f"Erro ao carregar ou processar arquivo CSV de Clima: {e}", file=sys.stderr)
        return pd.DataFrame()


# --- MAIN (Sem alterações) ---
def main():
    warnings.filterwarnings('ignore', category=FutureWarning)
    print("Iniciando data pipeline...")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Criado diretório '{DATA_DIR}'. Certifique-se de que seus CSVs estão lá.")

    # 1. Carregar dados
    ibge_df = load_ibge_from_csv(IBGE_PRODUCAO_FILE, region_filter=TARGET_REGION)
    cepea_df = load_cepea_from_csv(CEPEA_PRECOS_FILE)
    # Não passa mais o filtro de região, pois o arquivo não o suporta
    weather_df = load_weather_data(WEATHER_FILE)

    # 2. Unificar
    if cepea_df.empty or weather_df.empty:
        print("Erro: Não foi possível carregar dados de CEPEA ou Clima. Abortando.", file=sys.stderr)
        return

    print("Unificando dataframes...")
    unified_df = cepea_df.join(weather_df, how='outer')
    unified_df = unified_df.join(ibge_df, how='outer')

    unified_df = unified_df.sort_index()
    unified_df.index.name = 'date'

    # 3. Interpolar
    weather_cols = [col for col in weather_df.columns if col in unified_df.columns]
    unified_df[weather_cols] = unified_df[weather_cols].interpolate(method='time').fillna(0)

    ibge_cols = [col for col in ibge_df.columns if col in unified_df.columns]
    unified_df[ibge_cols] = unified_df[ibge_cols].ffill()

    # 4. Limpar
    main_price_col = 'preco_soja'
    if main_price_col not in unified_df.columns:
        print(f"Erro: Coluna principal '{main_price_col}' não encontrada após join.", file=sys.stderr)
        main_price_col = 'preco_milho'
        if main_price_col not in unified_df.columns:
            print("Nenhuma coluna de preço principal encontrada. Abortando.")
            return

    unified_df = unified_df.dropna(subset=[main_price_col])

    print(f"DataFrame unificado final criado com {len(unified_df)} registros.")
    print("\nColunas Finais:", unified_df.columns.tolist())
    print(unified_df.head())

    # 5. Salvar
    try:
        engine = create_engine(f'sqlite:///{DB_PATH}')
        unified_df.to_sql(DB_TABLE_NAME, engine, if_exists='replace', index=True)
        print(f"\nSucesso! Banco de dados unificado salvo em '{DB_PATH}' (Tabela: {DB_TABLE_NAME})")
    except ImportError:
        print("\nErro: A biblioteca 'sqlalchemy' não está instalada.", file=sys.stderr)
    except Exception as e:
        print(f"\nErro ao salvar banco de dados: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()