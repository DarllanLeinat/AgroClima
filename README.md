# 🌱 AgroClima - Inteligência Preditiva para o Agronegócio

Este projeto é uma plataforma de **Inteligência Preditiva** que integra dados históricos de preços (CEPEA), produção (IBGE) e variáveis meteorológicas para prever o mercado agrícola brasileiro. Utilizando Machine Learning, o sistema analisa como o clima impacta a produtividade e o valor de venda de commodities como Soja, Milho e Café.

## Como Funciona o Projeto

O sistema está dividido em três camadas principais:

1. **Pipeline de Dados (`data_pipeline.py`)**: Realiza a extração e limpeza de dados de arquivos CSV. Ele unifica séries temporais de preços, produção e clima em um banco de dados relacional **SQLite** via **SQLAlchemy**.
2. **Motor de Treinamento (`model_trainer.py`)**: Utiliza o algoritmo **Prophet** (Meta) para modelagem de séries temporais. O diferencial técnico é o uso de **regressores externos** (precipitação e temperatura), permitindo que o modelo aprenda a relação causal entre eventos climáticos e flutuações de mercado.
3. **Interface do Usuário (`app.py`)**: Um dashboard interativo construído em **Streamlit**. Ele permite que o produtor visualize projeções, intervalos de confiança e simule cenários de risco, como o impacto de secas no faturamento.

## Tecnologias Utilizadas

* **Linguagem:** Python.
* **Machine Learning:** Prophet.
* **Manipulação de Dados:** Pandas e NumPy.
* **Banco de Dados:** SQLite e SQLAlchemy.
* **Visualização:** Plotly e Streamlit.

## Pré-requisitos e Instalação

Para rodar este projeto localmente, você precisará ter o Python instalado (versão 3.8 ou superior recomendada).

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio

```

### 2. Instalar as dependências

```bash
pip install pandas numpy prophet plotly streamlit sqlalchemy

```

### 3. Preparar os dados

Antes de iniciar o dashboard, é necessário processar os dados e criar o banco de dados unificado:

```bash
python data_pipeline.py

```

### 4. Executar a aplicação

Após o processamento dos dados, inicie o dashboard:

```bash
streamlit run app.py

```

## Funcionalidades Técnicas

* **Decomposição de Séries:** Visualização detalhada de Tendência, Sazonalidade Anual e impacto de variáveis climáticas.
* **Simulação de Risco:** Slider interativo para reduzir a precipitação prevista e observar a resposta do modelo no preço e volume.
* **Intervalos de Confiança:** Gráficos que mostram a margem de erro estatística das previsões (yhat_lower e yhat_upper).

---
