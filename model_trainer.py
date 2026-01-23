import pandas as pd
import sqlite3
from prophet import Prophet
import sys
import os

DATA_DIR = 'data'
DB_NAME = 'agriforecast_unified_db.sqlite'
DB_PATH = os.path.join(DATA_DIR, DB_NAME)
TABLE_NAME = 'unified_data'

# Tenta encontrar a coluna de produção (ex: 'Produção, toneladas')
TARGET_COLUMN_NAME = 'produção'

def load_data_from_db():
    """Carrega os dados unificados do banco de dados SQLite."""
    # Usar DB_PATH (caminho completo)
    print(f"Carregando dados da tabela '{TABLE_NAME}' do banco '{DB_PATH}'...")

    if not os.path.exists(DB_PATH):
        print(f"Erro: Banco de dados '{DB_PATH}' não encontrado.", file=sys.stderr)
        print("Por favor, execute 'python data_pipeline.py' primeiro.", file=sys.stderr)
        return None

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(f'SELECT * FROM {TABLE_NAME}', conn,
                         parse_dates=['date'],
                         index_col='date')
        conn.close()

        if df.empty:
            print("Nenhum dado encontrado no banco de dados.", file=sys.stderr)
            return None

        print("Dados carregados com sucesso.")
        return df

    except Exception as e:
        print(f"Erro ao carregar dados do SQLite: {e}", file=sys.stderr)
        return None


def find_target_column(df, target_name):
    """Encontra a coluna de destino com base em um nome parcial."""
    for col in df.columns:
        if target_name.lower() in col.lower():
            print(f"Coluna alvo encontrada: '{col}'")
            return col
    return None


def train_prophet_model(df):
    """
    Treina um modelo Prophet básico para prever a coluna alvo.
    """

    target_column = find_target_column(df, TARGET_COLUMN_NAME)

    if not target_column:
        print(f"Erro: Nenhuma coluna contendo '{TARGET_COLUMN_NAME}' foi encontrada.", file=sys.stderr)
        print(f"Colunas disponíveis: {df.columns.tolist()}")
        return

    print(f"Treinando modelo Prophet para prever '{target_column}'...")

    # 1. Formata os dados para o Prophet (requer colunas 'ds' e 'y')
    df_prophet = df.reset_index()[['date', target_column]].rename(
        columns={'date': 'ds', target_column: 'y'}
    )

    # Remove linhas com 'y' nulo, se houver
    df_prophet.dropna(subset=['y'], inplace=True)

    if df_prophet.shape[0] < 2:
        print("Não há dados suficientes para treinar o Prophet (necessário > 1).", file=sys.stderr)
        return

    # 2. Adiciona regressores (outras variáveis)
    # Pega colunas de clima e preço, se existirem
    # ATUALIZADO: Adicionado 'temp_media' e 'precipitacao'
    regressors = [col for col in df.columns if (
            col.startswith('clima_') or
            col.startswith('preco_') or
            col.startswith('temp_') or  # Adicionado
            col.startswith('precip_') or  # Adicionado
            'Rendimento' in col or
            'Área' in col
    ) and col != target_column]

    if regressors:
        print(f"Adicionando regressores: {regressors}")
        df_prophet = df_prophet.merge(df.reset_index()[['date'] + regressors],
                                      left_on='ds', right_on='date').drop(columns='date')

    # 3. Inicializa e treina o modelo
    model = Prophet(yearly_seasonality=True)  # Adiciona sazonalidade anual
    for regressor in regressors:
        model.add_regressor(regressor)

    model.fit(df_prophet)

    # 4. Cria um DataFrame futuro para previsão

    # --- ATUALIZAÇÃO PARA CORRIGIR O WARNING ---
    # Mudei de 'Y' para 'YE' (Year-End)
    future = model.make_future_dataframe(periods=3, freq='YE')

    # Adiciona valores futuros para os regressores
    # Aqui, estamos apenas usando a média, mas o ideal seria prever os regressores também
    if regressors:
        # Pega a última linha de dados conhecidos para os regressores
        last_known_regressors = df_prophet[regressors].iloc[-1:].to_dict('records')[0]

        # Preenche os valores futuros com o último valor conhecido (ou média)
        # Preenchendo com o último valor (mais simples):
        for reg in regressors:
            if reg in future.columns:
                future[reg] = future[reg].fillna(last_known_regressors[reg])
            else:
                future[reg] = last_known_regressors[reg]

        # Copia valores históricos para garantir que não haja NaNs
        historical_part = df_prophet[['ds'] + regressors].set_index('ds')
        future = future.set_index('ds')
        future.update(historical_part)
        future = future.reset_index()

    # 5. Faz a previsão
    forecast = model.predict(future)

    # 6. Exibe e salva os resultados
    print("\n--- Previsão do Modelo Prophet ---")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5))

    try:
        fig = model.plot(forecast)
        plot_filename = 'forecast_plot.png'
        fig.savefig(plot_filename)
        print(f"\nGráfico de previsão salvo como '{plot_filename}'")

        fig_components = model.plot_components(forecast)
        components_filename = 'forecast_components_plot.png'
        fig_components.savefig(components_filename)
        print(f"Gráfico de componentes salvo como '{components_filename}'")

    except Exception as e:
        print(f"Não foi possível salvar os gráficos: {e}", file=sys.stderr)


def main():
    df = load_data_from_db()

    if df is not None:
        train_prophet_model(df)
        print("\nPara treinar um modelo LSTM, você precisará:")
        print("1. Normalizar os dados (ex: com MinMaxScaler do scikit-learn).")
        print("2. Criar sequências de 'janelas' de tempo para os dados de entrada.")
        print("3. Definir e treinar uma rede neural com camadas LSTM usando TensorFlow/Keras.")


if __name__ == "__main__":
    main()