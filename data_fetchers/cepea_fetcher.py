import pandas as pd
import sys
import os


def find_column_name(df, possible_names):
    """Função helper para encontrar um nome de coluna insensível a maiúsculas/minúsculas."""
    for col in df.columns:
        if col.lower() in possible_names:
            return col
    return None


def load_cepea_from_csv(file_path='data/cepea_soja_precos.csv', region_filter='Mato Grosso'):
    """
    Carrega dados de preços do CEPEA a partir de um arquivo CSV local.

    Esta versão ATUALIZADA agora filtra os dados por uma região específica.

    :param file_path: Caminho para o arquivo CSV.
    :param region_filter: A região específica para filtrar (ex: 'Mato Grosso').
    :return: DataFrame do Pandas com os dados de preço filtrados.
    """
    print(f"Carregando dados de preços do CEPEA de {file_path}...")

    if not os.path.exists(file_path):
        print(f"Erro: Arquivo CSV do CEPEA não encontrado em {file_path}", file=sys.stderr)
        print("Por favor, baixe os dados e salve-os como 'data/cepea_soja_precos.csv'", file=sys.stderr)
        return pd.DataFrame()

    try:
        # A codificação 'latin1' ou 'iso-8859-1' é comum em arquivos CSV brasileiros
        df = pd.read_csv(file_path,
                         encoding='latin1',
                         sep=';',
                         decimal=',',
                         thousands='.',
                         skiprows=3)  # Ajuste 'skiprows' conforme o arquivo

        # --- Lógica de Identificação de Coluna ---
        date_col = find_column_name(df, ['data', 'data_ref'])
        price_col = find_column_name(df, ['preço r$', 'preco r$', 'valor'])
        region_col = find_column_name(df, ['regiao', 'região', 'localidade', 'estado'])

        if not date_col or not price_col:
            print(f"Erro: Colunas de 'Data' ou 'Preço R$' não encontradas em {file_path}. Verifique o CSV.",
                  file=sys.stderr)
            return pd.DataFrame()

        print(f"Colunas identificadas: Data='{date_col}', Preço='{price_col}', Região='{region_col}'")

        # --- Lógica de Filtragem de Região (NOVO) ---
        if region_col:
            # Verifica se o filtro da região existe nos dados
            # Compara de forma insensível a maiúsculas/minúsculas
            df_region_lower = df[region_col].str.lower()
            region_filter_lower = region_filter.lower()

            if region_filter_lower in df_region_lower.unique():
                print(f"Filtrando dados para a região: '{region_filter}'")
                df = df[df_region_lower == region_filter_lower].copy()
            else:
                print(f"Aviso: Região '{region_filter}' não encontrada no CSV.", file=sys.stderr)
                print(f"Regiões encontradas: {df[region_col].unique()}", file=sys.stderr)
                print("Continuando sem filtro de preço para esta região.", file=sys.stderr)
                return pd.DataFrame()  # Retorna vazio se a região não for encontrada
        else:
            print("Aviso: Nenhuma coluna de 'Regiao' encontrada no CSV. Não é possível filtrar preços.",
                  file=sys.stderr)
            return pd.DataFrame()  # Retorna vazio se a coluna não existe

        # --- Limpeza e Formatação ---
        df['date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y')
        df.set_index('date', inplace=True)

        # Renomeia a coluna de preço para um nome padronizado
        df.rename(columns={price_col: 'preco_soja_brl'}, inplace=True)

        # Seleciona apenas a coluna de preço
        price_df = df[['preco_soja_brl']]

        print("Dados do CEPEA carregados e filtrados com sucesso.")
        return price_df

    except Exception as e:
        print(f"Erro ao carregar ou processar arquivo CSV do CEPEA: {e}", file=sys.stderr)
        return pd.DataFrame()


if __name__ == '__main__':
    # Teste rápido (requer que 'data/cepea_soja_precos.csv' exista)
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Criado diretório 'data/'. Por favor, adicione seu CSV do CEPEA lá.")
    else:
        # Testa o filtro
        cepea_data = load_cepea_from_csv(region_filter='Mato Grosso')
        if not cepea_data.empty:
            print("\n--- Dados Filtrados (Mato Grosso) ---")
            print(cepea_data.head())

