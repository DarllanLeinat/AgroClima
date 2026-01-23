AgriForecast Insights - Pipeline de Dados

Este projeto contém a base de código para o pipeline de dados do AgriForecast Insights. Ele é projetado para buscar, limpar e unificar dados agrícolas de múltiplas fontes brasileiras (IBGE, CEPEA, BR-DWGD, EMBRAPA) para treinar modelos de IA (Prophet, LSTM) para previsão de safras, preços e rendimento.

Estrutura do Projeto

streamlit run app.py

/data_fetchers: Contém módulos Python para buscar dados de APIs específicas.

ibge_fetcher.py: Busca dados de produção da API SIDRA do IBGE.

cepea_fetcher.py: Carrega dados de preços do CEPEA (atualmente de um CSV local).

weather_fetcher.py: Busca dados climáticos do Google Earth Engine.

embrapa_fetcher.py: Placeholder para a API da EMBRAPA (requer chave).

data_pipeline.py: O script orquestrador principal. Executa os fetchers, mescla os dados e os salva em agriforecast.db.

model_trainer.py: Um script de exemplo que carrega os dados do banco de dados e treina um modelo Prophet.

requirements.txt: As dependências Python.

Setup

Crie um Ambiente Virtual (Recomendado)

python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate


Instale as Dependências

pip install -r requirements.txt


Autentique o Google Earth Engine (CRUCIAL)
Para que o weather_fetcher.py funcione, você precisa se autenticar com o GEE:

earthengine-api authenticate


Siga as instruções no seu navegador para fazer login com uma conta do Google.

Obtenha Chaves de API (quando aplicável)

EMBRAPA: Registre-se no Portal EMBRAPA AgroAPI para obter uma chave de API. Adicione-a em data_fetchers/embrapa_fetcher.py.

Prepare os Dados Locais (CEPEA)

Crie um diretório chamado data/ na raiz do projeto.

Vá ao site do CEPEA e baixe a série histórica de preços (ex: soja).

IMPORTANTE: Salve seu arquivo como data/cepea_soja_precos.csv.

CERTIFIQUE-SE de que seu CSV tenha uma coluna de região (ex: Regiao ou Estado) para que o novo filtro funcione.

Como Executar

Execute o Pipeline de Dados
Este script buscará os dados de todas as fontes, irá uni-los (filtrando pela região "Mato Grosso") e salvar o resultado em um banco de dados SQLite chamado agriforecast.db.

python data_pipeline.py


Treine um Modelo de Exemplo
Após a execução do pipeline, você pode treinar um modelo Prophet de exemplo:

python model_trainer.py


Isso irá gerar um gráfico de previsão chamado forecast_plot.png.