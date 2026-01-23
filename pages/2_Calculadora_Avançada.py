import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import date, timedelta, datetime
import plotly.graph_objects as go
import requests
import json
from prophet.plot import plot_components_plotly
import locale  # <<< Para formatar meses em português
import calendar  # <<< Para obter nomes dos meses

# --- Configuração de Localização (Português) ---
# Isso é para garantir que os nomes dos meses (ex: "Janeiro") saiam corretamente
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, 'Portuguese_Brazil.1252')
    except locale.Error:
        st.warning(
            "Não foi possível definir o 'locale' para Português-Brasil. Os nomes dos meses podem aparecer em inglês.")

# --- Configuração (Copiado de app.py) ---
CULTURAS_SUPORTADAS = {
    "Soja (Soybean)": {
        "regioes": ["Mato Grosso (Soy)", "Paraná (Corn)"],
        "produtividade_padrao": 3.6,  # ton/ha
        "custo_semente_padrao": 450,
        "custo_fert_padrao": 1200,
        "custo_pest_padrao": 600,
        "custo_labor_padrao": 800,
    },
    "Milho (Corn)": {
        "regioes": ["Mato Grosso (Soy)", "Paraná (Corn)"],
        "produtividade_padrao": 6.0,  # ton/ha
        "custo_semente_padrao": 500,
        "custo_fert_padrao": 1300,
        "custo_pest_padrao": 500,
        "custo_labor_padrao": 700,
    },
    "Café Arábica (Arabica Coffee)": {
        "regioes": ["Minas Gerais (Coffee)"],
        "produtividade_padrao": 1.5,  # ton/ha
        "custo_semente_padrao": 300,
        "custo_fert_padrao": 1000,
        "custo_pest_padrao": 400,
        "custo_labor_padrao": 1500,
    }
}


# --- Funções de Simulação (CACHE-BUST) ---
@st.cache_data
def load_and_simulate_data_v5(crop, region):
    """Simulates historical price and weather data."""
    today = date.today()
    start_date = today - timedelta(days=5 * 365)
    if crop == "Soja (Soybean)":
        base_price = 150.00;
        volatility = 10.0;
        yearly_growth = 0.05
    elif crop == "Café Arábica (Arabica Coffee)":
        base_price = 800.00;
        volatility = 50.0;
        yearly_growth = 0.03
    else:  # Milho (Corn)
        base_price = 60.00;
        volatility = 5.0;
        yearly_growth = 0.04

    dates = pd.to_datetime(pd.date_range(start=start_date, end=today, freq='D'))
    n_days = len(dates)
    prices = base_price * (1 + yearly_growth * (dates.year - dates.year.min())) + \
             np.random.normal(0, volatility, n_days) + \
             volatility * np.sin(dates.dayofyear / 365 * 2 * np.pi)
    prices = np.maximum(prices, base_price * 0.8)
    df = pd.DataFrame({'ds': dates, 'y': prices.round(2)})
    avg_rainfall = 5 + 15 * np.sin(dates.dayofyear / 365 * 2 * np.pi) + np.random.normal(0, 3, n_days)
    df['rainfall_mm'] = np.maximum(avg_rainfall, 0).round(2)
    return df


@st.cache_data(show_spinner=False)
def run_prophet_forecast_v5_calc(df, periods=365):
    """Roda o Prophet APENAS para o preço (y)"""
    m_price = Prophet(
        seasonality_mode='multiplicative', changepoint_prior_scale=0.05,
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False
    )
    m_price.add_regressor('rainfall_mm')
    m_price.fit(df[['ds', 'y', 'rainfall_mm']])
    future = m_price.make_future_dataframe(periods=periods)
    future_dates = pd.to_datetime(future['ds'])
    future['rainfall_mm'] = (
            5 + 15 * np.sin(future_dates.dt.dayofyear / 365 * 2 * np.pi) +
            np.random.normal(0, 3, len(future))
    ).clip(lower=0).round(2)
    forecast_price = m_price.predict(future)
    return forecast_price


# --- Início da Página da Calculadora ---
st.set_page_config(layout="wide", page_title="Calculadora Avançada")

# CSS (Sem alterações)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    :root { --green-900: #0d5f49; --green-600: #2f8d6f; }
    .stApp { font-family: 'Inter', sans-serif; }
    .stMarkdown h2 { color: var(--green-900); font-weight: 700; text-align: center; border-bottom: 2px solid var(--green-600); padding-bottom: 10px; }
    .stMarkdown h3 { color: var(--green-900); font-weight: 600; }
    .stNumberInput > div > div, .stSelectbox > div > div { background-color: var(--secondary-background-color); border-radius: 8px; }
    .stButton > button { background-color: var(--green-900); color: white; font-weight: 700; padding: 10px 20px; border-radius: 8px; width: 100%; }
    .stButton > button:hover { background-color: var(--green-600); color: white; }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("## 🚜 Calculadora de Previsão Avançada")

# --- O FORMULÁRIO ---
with st.form(key="forecast_form"):
    st.markdown("### 1. Cultura e Localização")
    col1, col2 = st.columns(2)
    with col1:
        crop_choice = st.selectbox("Cultura *", options=CULTURAS_SUPORTADAS.keys(),
                                   help="Apenas as culturas que nosso modelo de IA suporta no momento.")
    with col2:
        region_choice = st.selectbox("Região *", options=CULTURAS_SUPORTADAS[crop_choice]["regioes"])

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Área plantada (ha) *", min_value=0.1, value=100.0, step=10.0)
    with col2:
        default_yield = CULTURAS_SUPORTADAS[crop_choice]["produtividade_padrao"]
        yield_hist = st.number_input("Produtividade estimada (ton/ha) *", min_value=0.1, value=default_yield, step=0.1,
                                     format="%.1f",
                                     help=f"Se não souber, use a média da região ({default_yield} ton/ha)")

    st.markdown("### 2. Custos de Produção (R$/ha)")
    col1, col2 = st.columns(2)
    with col1:
        seedCost = st.number_input("Custo da semente (R$/ha)", min_value=0,
                                   value=CULTURAS_SUPORTADAS[crop_choice]["custo_semente_padrao"])
    with col2:
        fertCost = st.number_input("Custo de fertilizante (R$/ha)", min_value=0,
                                   value=CULTURAS_SUPORTADAS[crop_choice]["custo_fert_padrao"])

    col1, col2 = st.columns(2)
    with col1:
        pestCost = st.number_input("Custo de defensivos (R$/ha)", min_value=0,
                                   value=CULTURAS_SUPORTADAS[crop_choice]["custo_pest_padrao"])
    with col2:
        laborCost = st.number_input("Mão de obra e outros (R$/ha)", min_value=0,
                                    value=CULTURAS_SUPORTADAS[crop_choice]["custo_labor_padrao"])

    st.markdown("### 3. Fatores Adicionais (Opcional)")
    col1, col2 = st.columns(2)
    with col1:
        soil = st.selectbox("Tipo de solo", ["Não informado", "Argiloso", "Arenoso", "Latossolo"])
    with col2:
        irrigation = st.selectbox("Sistema de irrigação", ["Sem irrigação", "Pivô central", "Gotejamento", "Aspersão"])

    submitted = st.form_submit_button("Gerar Previsão Avançada")

# --- LÓGICA DE CÁLCULO (FORA DO FORMULÁRIO) ---
if submitted:
    with st.spinner(f"Rodando IA para prever preços de {crop_choice}..."):
        df_hist = load_and_simulate_data_v5(crop_choice, region_choice)
        forecast = run_prophet_forecast_v5_calc(df_hist, periods=180)
        preco_futuro_tonelada = forecast[['yhat']].iloc[-1].values[0]

    st.markdown("---")
    st.markdown("## 📊 Resultado da Previsão")

    producao_total = area * yield_hist
    receita_bruta = producao_total * preco_futuro_tonelada
    custo_total_por_ha = seedCost + fertCost + pestCost + laborCost
    custo_total = area * custo_total_por_ha
    lucro_liquido = receita_bruta - custo_total

    st.success(f"### Lucro Líquido Projetado: R$ {lucro_liquido:,.2f}")

    # --- MODELAGEM st.metric (JÁ ESTAVA AQUI) ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Receita Bruta Estimada", f"R$ {receita_bruta:,.2f}",
                f"{producao_total:,.1f} ton @ R$ {preco_futuro_tonelada:,.2f}/ton")
    col2.metric("Custo Total Estimado", f"R$ {custo_total:,.2f}", f"R$ {custo_total_por_ha:,.2f} / ha")
    col3.metric("Preço Futuro (180 dias)", f"R$ {preco_futuro_tonelada:,.2f} / ton")
    # --- FIM DO st.metric ---

    st.markdown("---")
    st.markdown("### 💡 Insights & Recomendações")
    if receita_bruta <= 0:
        st.error("**Alerta de Risco:** Receita Bruta é zero ou negativa. Verifique seus inputs.")
    elif (lucro_liquido / receita_bruta) < 0.15:
        st.error(
            f"**Alerta de Risco:** Sua margem de lucro projetada está muito baixa ({lucro_liquido / receita_bruta:.1%}). Considere revisar seus custos.")
    if irrigation == "Sem irrigação":
        st.warning(
            f"**Risco Climático:** Seu plano de {crop_choice} sem irrigação é altamente dependente da previsão de chuvas.")
    else:
        st.success(f"**Vantagem:** O uso de {irrigation} estabiliza sua produtividade ({yield_hist} ton/ha).")

    st.markdown("---")
    st.markdown(f"### 📈 Previsão da Tendência de Preço (R$/ton) - Próximos 180 dias")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão de Preço',
                                  line=dict(color='#0d5f49', width=3)))
    fig_line.add_trace(go.Scatter(
        x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
        y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(13, 95, 73, 0.2)', line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", showlegend=False, name='Intervalo de Confiança'
    ))
    fig_line.update_layout(title_text=f"Previsão do Preço (R$/ton) para {crop_choice}", xaxis_title="Data",
                           yaxis_title="Preço (R$)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_line, use_container_width=True)


# ============================================================
# 🌦 SEÇÃO EXTRA — CLIMA REAL (OPEN-METEO)
# ============================================================

@st.cache_data(ttl=3600)  # Cache por 1 hora
def get_coordinates_for_city(city_name):
    """Busca as coordenadas (lat, lon) de uma cidade usando Open-Meteo Geocoding."""
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": city_name, "count": 1, "language": "pt", "format": "json"}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "results" not in data or len(data["results"]) == 0:
            return None, f"Cidade '{city_name}' não encontrada."
        result = data["results"][0]
        lat, lon = result["latitude"], result["longitude"]
        display_name = result.get("name", city_name)
        if "admin1" in result:
            display_name += f" ({result['admin1']})"
        return (lat, lon), f"Localização encontrada: {display_name}"
    except requests.exceptions.RequestException as e:
        return None, f"Erro de rede ao buscar coordenadas: {e}"
    except Exception as e:
        return None, f"Erro ao processar dados de geocoding: {e}"


@st.cache_data(ttl=600)  # Cache de dados do clima por 10 minutos
def get_weather_from_open_meteo(lat, lon, days_past):
    """Busca dados climáticos históricos da Open-Meteo."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon, "past_days": days_past,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "America/Sao_Paulo"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "daily" not in data:
            return None, "A API não retornou dados diários."
        df_clima = pd.DataFrame(data["daily"])
        df_clima = df_clima.rename(columns={
            "time": "data", "temperature_2m_max": "temp_max",
            "temperature_2m_min": "temp_min", "precipitation_sum": "chuva_mm"
        })
        df_clima["data"] = pd.to_datetime(df_clima["data"])
        return df_clima, "Dados carregados com sucesso!"
    except requests.exceptions.RequestException as e:
        return None, f"Erro de rede ao buscar dados do clima: {e}"
    except Exception as e:
        return None, f"Erro ao processar dados do clima: {e}"


st.markdown("---")
st.markdown("## 🌦 Consultar Clima Real (Open-Meteo)")
with st.expander("Clique para abrir a consulta de clima real"):
    st.markdown("Esta ferramenta consulta **dados climáticos históricos** usando a API gratuita Open-Meteo.")
    cidade_nome = st.text_input("Nome da Cidade", "Cuiabá", help="Exemplo: Cuiabá, Rondonópolis, Londrina, Sinop")
    dias = st.slider("Quantos dias no passado deseja consultar?", 1, 90, 7, key="openmeteo_dias_slider")
    if st.button("Buscar Clima Real"):
        with st.spinner(f"Procurando coordenadas para '{cidade_nome}'..."):
            coords, status_msg = get_coordinates_for_city(cidade_nome)
        if coords is None:
            st.error(status_msg)
        else:
            st.info(status_msg)
            lat, lon = coords
            with st.spinner(f"Consultando dados do clima para {cidade_nome} (últimos {dias} dias)..."):
                df_clima, status_clima = get_weather_from_open_meteo(lat, lon, dias)
            if df_clima is None:
                st.error(status_clima)
            else:
                st.success(status_clima)
                st.dataframe(df_clima)
                # Gráfico de chuva
                fig_chuva = go.Figure()
                fig_chuva.add_trace(
                    go.Scatter(x=df_clima["data"], y=df_clima["chuva_mm"], mode="lines+markers", name="Chuva (mm)",
                               line=dict(width=3, color='#0077b6')))
                fig_chuva.update_layout(title="Precipitação (Chuva em mm)", xaxis_title="Data",
                                        yaxis_title="Chuva (mm)")
                st.plotly_chart(fig_chuva, use_container_width=True)
                # Gráfico de temperatura
                fig_temp = go.Figure()
                fig_temp.add_trace(
                    go.Scatter(x=df_clima["data"], y=df_clima["temp_max"], mode="lines+markers", name="Temperatura Máx",
                               line=dict(color='#d62728')))
                fig_temp.add_trace(
                    go.Scatter(x=df_clima["data"], y=df_clima["temp_min"], mode="lines+markers", name="Temperatura Mín",
                               line=dict(color='#1f77b4')))
                fig_temp.update_layout(title="Temperatura (°C)", xaxis_title="Data", yaxis_title="Temperatura")
                st.plotly_chart(fig_temp, use_container_width=True)


# ============================================================
# 🤖 ANALISADOR IA CSV
# ============================================================

def processar_csv_ibge(df):
    """
    Processa um CSV no formato específico do IBGE.
    *** VERSÃO CORRIGIDA (procura colunas e meses em minúsculas) ***
    """
    try:
        # Mapa de meses agora em minúsculas
        meses_map = {'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4, 'maio': 5, 'junho': 6, 'julho': 7,
                     'agosto': 8, 'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12}

        # Garante que as colunas 'mes' e 'ano' são strings antes de mapear
        # (O script principal já deve ter feito o .lower() nas colunas)

        # --- LINHA CORRIGIDA 1 ---
        # Adicionamos .str.lower() para forçar o CONTEÚDO da coluna 'mes' a ser minúsculo
        df['mes'] = df['mes'].astype(str).str.lower()

        df['ano'] = df['ano'].astype(str)
        df['dia'] = df['dia'].astype(str)

        df['mes_num'] = df['mes'].map(meses_map)

        df['ds'] = pd.to_datetime(
            df[['ano', 'mes_num', 'dia']].rename(columns={'ano': 'year', 'mes_num': 'month', 'dia': 'day'}),
            errors='coerce'  # Adiciona 'errors=coerce' por segurança, para criar NaT se a data for inválida
        )

        # Procura 'preco (r$/saca)' (minúsculo) e 'quantidade (toneladas)'
        df = df.rename(columns={'preco (r$/saca)': 'y', 'quantidade (toneladas)': 'volume'})

        # Converte 'y' e 'volume' para numérico, tratando erros
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        df['sales_value'] = df['y'] * (df['volume'] * (1000 / 60))  # Assumindo saca de 60kg
        df_processado = df[['ds', 'produto', 'regiao', 'y', 'volume', 'sales_value']]

        # --- LINHA CORRIGIDA 2 ---
        # Adicionamos 'ds' ao subset do dropna()
        # Isso remove qualquer linha que tenha data, preço OU volume inválidos.
        df_processado = df_processado.dropna(subset=['ds', 'y', 'volume'])

        if df_processado.empty:
            st.error(
                "Erro ao processar: Nenhum dado válido (numérico) encontrado para 'preco' ou 'quantidade' após a conversão, ou as datas são inválidas.")
            return None

        st.success("Ficheiro do IBGE processado com sucesso!")
        return df_processado
    except Exception as e:
        st.error(f"Erro ao processar o ficheiro do IBGE: {e}.")
        return None


@st.cache_data(show_spinner=False)
def run_prophet_forecast_v5_csv(df, periods=365):
    """
    Roda o Prophet para prever 'y' (PREÇO) e (opcionalmente) 'volume'.
    Retorna os dataframes de componentes 'forecast_price' e 'forecast_volume'.
    """
    required_cols = ['ds', 'y']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Erro no CSV! O CSV precisa ter no mínimo as colunas: {', '.join(required_cols)}")
        return None, None, None, None, None

        # --- CORREÇÃO NaN ---
    # Garante que não há NaNs na coluna 'ds' antes de passar ao Prophet
    df = df.dropna(subset=['ds'])
    if df.empty:
        st.error("Erro: Após limpar datas inválidas, nenhum dado restou para a análise.")
        return None, None, None, None, None

    has_volume = 'volume' in df.columns
    has_rainfall = 'rainfall_mm' in df.columns

    # --- 1. Treinar Modelo de Preço ('y') ---
    m_price = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.05, yearly_seasonality=True,
                      weekly_seasonality=False, daily_seasonality=False)
    df_price_fit = df[['ds', 'y']].copy().dropna(subset=['y'])

    if has_rainfall:
        m_price.add_regressor('rainfall_mm')
        df_price_fit['rainfall_mm'] = df['rainfall_mm']

    m_price.fit(df_price_fit)
    future = m_price.make_future_dataframe(periods=periods)

    if has_rainfall:
        if len(df) > 365:
            rainfall_mean = df['rainfall_mm'].mean()
            rainfall_amplitude = (df['rainfall_mm'].max() - df['rainfall_mm'].min()) / 2
            future['rainfall_mm'] = rainfall_mean + rainfall_amplitude * np.sin(future.dt.dayofyear / 365 * 2 * np.pi)
        else:
            future['rainfall_mm'] = 5 + 15 * np.sin(future.dt.dayofyear / 365 * 2 * np.pi)
        future['rainfall_mm'] = future['rainfall_mm'].clip(lower=0).round(2)

    forecast_price = m_price.predict(future)

    # --- 2. Treinar Modelo de Volume ('volume') ---
    m_volume = None
    forecast_volume = None
    final_forecast = forecast_price.copy()

    if has_volume:
        st.success("Coluna 'volume' (Produção) detetada! Treinando o modelo de produção...")
        df_volume_fit = df[['ds', 'volume']].copy().rename(columns={'volume': 'y'})
        df_volume_fit = df_volume_fit.dropna(subset=['y'])

        m_volume = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.05, yearly_seasonality=True,
                           weekly_seasonality=False, daily_seasonality=False)

        if has_rainfall:
            m_volume.add_regressor('rainfall_mm')
            df_volume_fit['rainfall_mm'] = df['rainfall_mm']

        if len(df_volume_fit) > 2:
            m_volume.fit(df_volume_fit)
            forecast_volume = m_volume.predict(future)
            final_forecast['volume_forecast'] = forecast_volume['yhat']
            final_forecast['volume_lower'] = forecast_volume['yhat_lower']
            final_forecast['volume_upper'] = forecast_volume['yhat_upper']
            final_forecast['sales_value_forecast'] = final_forecast['yhat'] * final_forecast['volume_forecast']
        else:
            st.warning("Não há dados de 'volume' suficientes para treinar o modelo de produção.")
            m_volume = None

    return m_price, m_volume, final_forecast, forecast_price, forecast_volume


# --- FUNÇÃO DE ESTILO E TRADUÇÃO (ATUALIZADA) ---
def melhorar_grafico_componentes(fig):
    """
    Aprimora o gráfico de componentes do Prophet:
    1. Traduz os títulos e legendas.
    2. Aplica um estilo visual mais limpo e profissional.
    """

    # Mapeamento de traduções e estilos
    trace_map = {
        'trend': {'name': 'Tendência', 'color': '#0d5f49', 'width': 3},  # Cor principal (verde escuro)
        'yearly': {'name': 'Sazonalidade Anual', 'color': '#ff7f0e', 'width': 2},  # Laranja
        'rainfall_mm': {'name': 'Impacto Chuva (mm)', 'color': '#1f77b4', 'width': 2}  # Azul
    }

    # 1. Atualizar Traces (Linhas)
    for trace in fig.data:
        original_name = trace.name.split('_')[0] if '_' in trace.name else trace.name

        if original_name in trace_map:
            config = trace_map[original_name]

            if trace.name == original_name:
                trace.name = config['name']
                trace.line.color = config['color']
                trace.line.width = config['width']

            if 'upper' in trace.name or 'lower' in trace.name:
                trace.line.color = 'rgba(255,255,255,0)'
                trace.showlegend = False

                main_color_hex = config['color'].lstrip('#')
                rgb = tuple(int(main_color_hex[i:i + 2], 16) for i in (0, 2, 4))
                trace.fillcolor = f'rgba({rgb[0]},{rgb[1]},{rgb[2]},0.2)'

        if 'Changepoints' in trace.name:
            trace.marker.color = '#d62728'  # Vermelho
            trace.marker.size = 4
            trace.name = 'Pontos de Mudança'

    # 2. Atualizar Layout (Geral)
    fig.update_layout(
        font_family="Inter",
        font_color="#333333",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font_size=12
        )
    )

    # 3. Atualizar Anotações (Títulos dos Subplots) e Eixos
    annotation_map = {
        'trend': 'Tendência',
        'yearly': 'Sazonalidade Anual',
        'rainfall_mm': 'Impacto da Chuva (mm)'
    }

    for ann in fig.layout.annotations:
        if ann.text in annotation_map:
            ann.text = annotation_map[ann.text]  # Traduz

        ann.font_size = 14
        ann.font_color = "#0d5f49"  # Verde escuro
        ann.xanchor = 'center'
        ann.x = 0.5  # Centraliza o título

    # 4. Atualizar Eixos
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(230, 230, 230, 0.7)', zeroline=False)

    return fig


# ============================================================
# 💡💡💡 NOVA FUNÇÃO DE ANÁLISE AUTOMÁTICA 💡💡💡
# ============================================================

def gerar_analise_textual_modelo(model, forecast_df, historical_df, nome_metrica="Dados"):
    """
    Gera uma análise textual automática (insights) a partir de um modelo Prophet
    e seus dados de previsão e históricos.

    Args:
        model (Prophet): O modelo Prophet treinado.
        forecast_df (pd.DataFrame): O DataFrame de previsão (output do model.predict()).
        historical_df (pd.DataFrame): O DataFrame histórico (input, deve ter 'ds' e 'y').
        nome_metrica (str): O nome da métrica sendo analisada (ex: "Preço", "Produção").

    Returns:
        list: Uma lista de strings, onde cada string é um insight.
    """
    insights = []

    # Garante que 'y' seja numérico no histórico
    historical_df['y'] = pd.to_numeric(historical_df['y'], errors='coerce')
    historical_df = historical_df.dropna(subset=['y'])

    if historical_df.empty:
        return ["Não há dados históricos suficientes para a análise textual."]

    # 1. Análise de Tendência (Trend)
    try:
        # Pega a tendência apenas no período histórico
        historical_trend = forecast_df.loc[forecast_df['ds'] <= historical_df['ds'].max(), 'trend']
        if not historical_trend.empty:
            trend_start = historical_trend.iloc[0]
            trend_end = historical_trend.iloc[-1]

            # Evita divisão por zero se a tendência começar em 0
            if trend_start != 0:
                percent_change = ((trend_end / trend_start) - 1) * 100
                if percent_change > 5:
                    insights.append(
                        f"**Tendência (Geral):** A sua **{nome_metrica}** histórica mostra uma **tendência de alta** de longo prazo, com um crescimento estimado de **{percent_change:+.1f}%** no período analisado.")
                elif percent_change < -5:
                    insights.append(
                        f"**Tendência (Geral):** A sua **{nome_metrica}** histórica mostra uma **tendência de baixa** de longo prazo, com uma queda estimada de **{percent_change:+.1f}%** no período analisado.")
                else:
                    insights.append(
                        f"**Tendência (Geral):** A sua **{nome_metrica}** histórica manteve-se **relativamente estável**, com uma variação de **{percent_change:+.1f}%** no período analisado.")
            else:
                insights.append(
                    f"**Tendência (Geral):** A tendência da sua **{nome_metrica}** iniciou em zero, terminando em **{trend_end:,.2f}**.")
    except Exception as e:
        insights.append(f"Não foi possível calcular a tendência: {e}")

    # 2. Análise de Sazonalidade (Yearly)
    try:
        if 'yearly' in forecast_df.columns:
            yearly_data = forecast_df[['ds', 'yearly']].copy()
            yearly_data['month'] = yearly_data['ds'].dt.month

            # Agrupa pela média do impacto mensal
            avg_monthly_seasonality = yearly_data.groupby('month')['yearly'].mean()

            pico_mes_num = avg_monthly_seasonality.idxmax()
            vale_mes_num = avg_monthly_seasonality.idxmin()

            # Converte número do mês para nome em português
            pico_mes_nome = calendar.month_name[pico_mes_num].capitalize()
            vale_mes_nome = calendar.month_name[vale_mes_num].capitalize()

            insights.append(
                f"**Sazonalidade (Padrão Anual):** O modelo aprendeu que a sua **{nome_metrica}** é **naturalmente mais forte em {pico_mes_nome}** (pico da sazonalidade) e **mais fraca em {vale_mes_nome}** (vale da sazonalidade).")
    except Exception as e:
        insights.append(f"Não foi possível calcular a sazonalidade: {e}")

    # 3. Análise de Erros (Resíduos)
    try:
        # Compara o real (y) com o previsto (yhat) no período histórico
        df_merged = pd.merge(historical_df[['ds', 'y']], forecast_df[['ds', 'yhat']], on='ds')

        if not df_merged.empty:
            df_merged['residual'] = df_merged['y'] - df_merged['yhat']
            df_merged['residual_abs'] = df_merged['residual'].abs()

            # Calcula o MAE (Mean Absolute Error)
            mae = df_merged['residual_abs'].mean()
            mean_value = historical_df['y'].mean()

            if mean_value != 0:
                mae_percent = (mae / mean_value) * 100
                insights.append(
                    f"**Precisão do Modelo (Erro):** Em média, o modelo errou a **{nome_metrica}** em **{mae:,.2f}** unidades (MAE), o que representa um erro percentual médio de **{mae_percent:.1f}%** em relação à média histórica.")
            else:
                insights.append(
                    f"**Precisão do Modelo (Erro):** Em média, o modelo errou a **{nome_metrica}** em **{mae:,.2f}** unidades (MAE).")

            # Encontra o pior dia
            maior_erro_idx = df_merged['residual_abs'].idxmax()
            maior_erro_data = df_merged.loc[maior_erro_idx, 'ds'].strftime('%d/%m/%Y')
            maior_erro_real = df_merged.loc[maior_erro_idx, 'y']
            maior_erro_prev = df_merged.loc[maior_erro_idx, 'yhat']

            insights.append(
                f"**Maior Erro:** O modelo teve sua **maior dificuldade** em **{maior_erro_data}**, quando o valor real foi **{maior_erro_real:,.2f}** e o modelo previu **{maior_erro_prev:,.2f}**. (Diferença de {df_merged.loc[maior_erro_idx, 'residual']:,.2f}).")
    except Exception as e:
        insights.append(f"Não foi possível calcular os erros do modelo: {e}")

    return insights


# --- FIM DA NOVA FUNÇÃO ---


# --- Início da Página (Analisador CSV) ---
st.markdown("## 🤖 Analisador de CSV com IA (Prophet)")
st.markdown("---")
st.markdown("### 1. Carregue seus Dados Históricos")
st.info(
    """
    Arraste o seu ficheiro CSV para cá. A IA irá tentar detetar o formato.
    **Formato Padrão (Recomendado):** `ds` (data, YYYY-MM-DD), `y` (preço), `volume` (produção).
    **Formato IBGE (O seu `ibge_producao_mt.csv`):** `dia`, `mes`, `ano`, `produto`, `quantidade (toneladas)`, `preco (R$/saca)`.
    """
)

uploaded_file = st.file_uploader("Arraste seu arquivo .csv para cá", type="csv", key="csv_uploader")

# --- Bloco TRY/EXCEPT principal para todo o processamento do arquivo ---
if uploaded_file is not None:
    try:
        # --- Bloco de Leitura do CSV (com try/except aninhado) ---
        try:
            df_hist_raw = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df_hist_raw = pd.read_csv(uploaded_file, encoding='latin1')
                st.info("Arquivo lido com codificação 'latin1'.")
            except Exception as e:
                st.error(f"Não foi possível ler o CSV. Tente salvar em UTF-8. Erro: {e}")
                st.stop()
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}.")
            st.stop()

        # --- Bloco de Processamento de Colunas ---
        df_hist = df_hist_raw.copy()
        # Garante que colunas sejam strings e minúsculas
        colunas = [str(c).lower() for c in df_hist.columns]
        df_hist.columns = colunas  # Renomeia colunas para minúsculas

        if 'dia' in colunas and 'mes' in colunas and 'ano' in colunas and 'produto' in colunas:
            st.warning("Detetado um ficheiro em formato IBGE.")
            df_hist = processar_csv_ibge(df_hist)  # Chama a função CORRIGIDA
            if df_hist is not None:
                produtos_disponiveis = df_hist['produto'].unique()
                produto_selecionado = st.selectbox("Selecione o Produto para Analisar:", produtos_disponiveis)
                df_analise = df_hist[df_hist['produto'] == produto_selecionado].copy()
                previsao_label = f"Previsão de {produto_selecionado}"
            else:
                st.error("O processamento do ficheiro IBGE falhou. Verifique os dados e o formato.")
                st.stop()
        elif 'ds' in colunas and 'y' in colunas:
            st.success("Detetado um ficheiro em formato Padrão.")
            df_hist['ds'] = pd.to_datetime(df_hist['ds'], errors='coerce')
            df_hist = df_hist.dropna(subset=['ds'])  # Remove datas inválidas
            df_analise = df_hist
            previsao_label = "Previsão de 'y'"
        else:
            st.error("Erro no CSV! Não foi possível detetar as colunas necessárias ('ds' e 'y', ou o formato IBGE).")
            st.stop()

        # --- Bloco de Interface (dentro do try principal) ---
        st.markdown("### Tabela de Dados (Amostra Processada)")
        st.dataframe(df_analise.head())
        st.markdown("---")
        st.markdown("### 2. Configure e Rode a Análise da IA")
        periods = st.slider(
            "Horizonte da Previsão (dias no futuro):",
            min_value=30, max_value=730, value=365, step=30,
            help="Quantos dias no futuro você quer que a IA preveja?", key="csv_slider_periods"
        )

        if st.button("Processar com IA (Prophet)", key="csv_process_button"):
            with st.spinner(f"Rodando IA nos seus dados para prever {periods} dias..."):

                m_price, m_volume, forecast, forecast_price_comp, forecast_volume_comp = run_prophet_forecast_v5_csv(
                    df_analise, periods=periods)

                if forecast is not None and forecast_price_comp is not None:
                    st.markdown("---")
                    st.markdown("## 📊 Resultados da Análise da IA")
                    st.markdown("### Tabela: Previsão (Últimos dias)")

                    if m_volume:
                        st.dataframe(forecast[['ds', 'yhat', 'volume_forecast', 'sales_value_forecast']].tail())
                    else:
                        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

                    # --- SEÇÃO DE ANÁLISE AUTOMÁTICA (COM SEUS PRÓPRIOS try/except) ---
                    st.markdown("---")
                    st.markdown("### 💡 Análise Automática da IA")
                    st.info("Abaixo está uma análise gerada pela IA sobre o que o modelo aprendeu com seus dados.")

                    # --- Análise de Preço ---
                    st.markdown("#### Análise de PREÇO ('y')")
                    try:
                        insights_preco = gerar_analise_textual_modelo(m_price, forecast_price_comp, df_analise, "Preço")
                        for insight in insights_preco:
                            st.markdown(f"- {insight}")
                    except Exception as e:
                        st.warning(f"Não foi possível gerar a análise textual do preço: {e}")

                    # --- Análise de Volume (se existir) ---
                    if m_volume and forecast_volume_comp is not None:
                        st.markdown("#### Análise de PRODUÇÃO ('volume')")
                        try:
                            # Prepara os dados de volume para a função (ela espera colunas 'y' e 'yhat')
                            df_analise_volume = df_analise[['ds', 'volume']].rename(columns={'volume': 'y'}).dropna(
                                subset=['y'])

                            insights_volume = gerar_analise_textual_modelo(m_volume, forecast_volume_comp,
                                                                           df_analise_volume, "Produção (Volume)")
                            for insight in insights_volume:
                                st.markdown(f"- {insight}")
                        except Exception as e:
                            st.warning(f"Não foi possível gerar a análise textual da produção: {e}")

                    # --- FIM DA SEÇÃO DE ANÁLISE ---

                    # --- Bloco de Gráficos ---
                    if m_volume:
                        st.markdown("---")
                        st.markdown(f"### 📈 Gráfico: Análise de Vendas e Preços - {previsao_label}")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_analise['ds'], y=df_analise['sales_value'], mode='markers+lines',
                                                 line=dict(color='#1f77b4', width=3), marker=dict(size=4, opacity=0.8),
                                                 name='Valor de Vendas Real', yaxis='y'))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['sales_value_forecast'], mode='lines',
                                                 line=dict(color='#1f77b4', width=2, dash='dash'),
                                                 name='Valor de Vendas Previsto', yaxis='y'))
                        fig.add_trace(go.Scatter(x=df_analise['ds'], y=df_analise['y'], mode='markers+lines',
                                                 line=dict(color='#ff7f0e', width=3), marker=dict(size=4, opacity=0.8),
                                                 name='Preço Real (R$/saca)', yaxis='y2'))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines',
                                                 line=dict(color='#ff7f0e', width=2, dash='dash'),
                                                 name='Preço Previsto (R$/saca)', yaxis='y2'))
                        fig.update_layout(
                            title=dict(text=f'🌱 Análise de Vendas e Preços - {previsao_label}', x=0.5,
                                       font=dict(size=20, color='#0d5f49')),
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                                        font=dict(size=12)),
                            height=500, showlegend=True, plot_bgcolor='rgba(242, 247, 243, 0.3)', paper_bgcolor='white',
                            yaxis=dict(title="Valor de Vendas (R$)", tickformat=',.0f', tickprefix="R$ ",
                                       title_font=dict(color='#1f77b4'), tickfont=dict(color='#1f77b4')),
                            yaxis2=dict(title="Preço (R$/saca)", tickformat=',.2f', tickprefix="R$ ",
                                        title_font=dict(color='#ff7f0e'), tickfont=dict(color='#ff7f0e'),
                                        overlaying='y', side='right')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.markdown("---")
                        st.markdown(f"### 📈 Gráfico: {previsao_label}")
                        fig_line_price = go.Figure()
                        fig_line_price.add_trace(
                            go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão',
                                       line=dict(color='#0d5f49', width=3)))
                        fig_line_price.add_trace(go.Scatter(x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                                                            y=pd.concat(
                                                                [forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                                                            fill='toself', fillcolor='rgba(13, 95, 73, 0.2)',
                                                            line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip",
                                                            showlegend=False))
                        fig_line_price.add_trace(
                            go.Scatter(x=df_analise['ds'], y=df_analise['y'], mode='markers', name='Dados Históricos',
                                       marker=dict(color='#ff7f0e', size=4)))
                        fig_line_price.update_layout(xaxis_title="Data", yaxis_title="Valor Previsto",
                                                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_line_price, use_container_width=True)

                    # --- GRÁFICOS DE COMPONENTES ---
                    st.markdown("---")
                    st.markdown("### 🧩 Decomposição da Previsão (Preço)")
                    st.info(
                        "Isto mostra *por que* o seu preço está a mudar, com base na Tendência, Sazonalidade (padrões anuais) e Clima (se disponível).")

                    fig_comp_price = plot_components_plotly(m_price, forecast_price_comp)
                    fig_comp_price = melhorar_grafico_componentes(fig_comp_price)
                    st.plotly_chart(fig_comp_price, use_container_width=True)

                    if m_volume and forecast_volume_comp is not None:
                        st.markdown("---")
                        st.markdown("### 🧩 Decomposição da Previsão (Produção)")
                        st.info(
                            "Isto mostra *por que* a sua produção está a mudar, com base na Tendência, Sazonalidade e Clima (se disponível).")

                        fig_comp_volume = plot_components_plotly(m_volume, forecast_volume_comp)
                        fig_comp_volume = melhorar_grafico_componentes(
                            fig_comp_volume)
                        st.plotly_chart(fig_comp_volume, use_container_width=True)

                else:
                    st.error("Ocorreu um erro ao rodar a previsão da IA. Verifique os dados de entrada.")

    # --- Bloco EXCEPT final (deve estar alinhado com o 'try' principal lá de cima) ---
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}. Verifique o formato do seu CSV.")