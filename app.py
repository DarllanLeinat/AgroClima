import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import date, timedelta
from prophet.plot import plot_components_plotly

# --- Configuração (EDITADO AQUI) ---
WEATHER_LAT_LON = {
    "Mato Grosso": (-12.6657, -55.9904),  # Antigo: "Mato Grosso (Soy)"
    "Minas Gerais": (-20.7515, -46.5658),  # Antigo: "Minas Gerais (Coffee)"
    "Paraná": (-24.8988, -52.3768)  # Antigo: "Paraná (Corn)"
}


# --- FIM DA EDIÇÃO ---


# --- 1. Funções de Simulação de Dados (CACHE ATUALIZADO) ---

@st.cache_data
def load_and_simulate_data_v5(crop, region):  # <--- MUDANÇA DE NOME (v5)
    """Simulates historical price, volume, and weather data."""
    today = date.today()
    start_date = today - timedelta(days=5 * 365)  # 5 years of data

    produtividade_padrao = 3.6  # Default

    if crop == "Soja (Soybean)":
        base_price = 150.00
        volatility = 10.0
        yearly_growth = 0.05
        base_volume = 1000  # tons
        volume_volatility = 100
        produtividade_padrao = 3.6  # ton/ha (aprox 60 sacas)
    elif crop == "Café Arábica (Arabica Coffee)":
        base_price = 800.00
        volatility = 50.0
        yearly_growth = 0.03
        base_volume = 500  # tons
        volume_volatility = 50
        produtividade_padrao = 1.5  # ton/ha (aprox 25 sacas)
    else:  # Milho (Corn)
        base_price = 60.00
        volatility = 5.0
        yearly_growth = 0.04
        base_volume = 2000  # tons
        volume_volatility = 200
        produtividade_padrao = 6.0  # ton/ha (aprox 100 sacas)

    dates = pd.to_datetime(pd.date_range(start=start_date, end=today, freq='D'))
    n_days = len(dates)

    prices = base_price * (1 + yearly_growth * (dates.year - dates.year.min())) + \
             np.random.normal(0, volatility, n_days) + \
             volatility * np.sin(dates.dayofyear / 365 * 2 * np.pi)
    prices = np.maximum(prices, base_price * 0.8)

    volumes = base_volume + \
              np.random.normal(0, volume_volatility, n_days) + \
              volume_volatility * 0.5 * np.sin(dates.dayofyear / 365 * 2 * np.pi + np.pi / 4)
    volumes = np.maximum(volumes, base_volume * 0.5)

    sales_value = prices * volumes
    df = pd.DataFrame({
        'ds': dates, 'y': prices.round(2),
        'volume': volumes.round(0), 'sales_value': sales_value.round(2)
    })
    avg_rainfall = 5 + 15 * np.sin(dates.dayofyear / 365 * 2 * np.pi) + np.random.normal(0, 3, n_days)
    df['rainfall_mm'] = np.maximum(avg_rainfall, 0).round(2)

    # Retorna o df E o valor padrão de produtividade
    return df, produtividade_padrao


@st.cache_data
def get_current_weather_v5(region):  # <--- MUDANÇA DE NOME (v5)
    """Simulates real-time weather data."""
    temp = np.random.randint(20, 35);
    precip = np.random.choice([0.0, 2.5, 10.0])
    condition = np.random.choice(["Céu Limpo", "Nublado", "Chuva Forte"])
    return {"Temperatura Atual": f"{temp}°C", "Precipitação (24h)": f"{precip} mm", "Condição": condition}


# --- 2. Modelo Prophet (CORRIGIDO e com CACHE-BUST) ---
@st.cache_data(show_spinner=False)
def run_prophet_forecast_v5(df, periods=365):  # <--- MUDANÇA DE NOME (v5)
    """Runs the Prophet model to forecast prices, volumes, and sales values."""

    # --- CORREÇÃO ROBUSTA DO BUG (ValueError) ---
    # Verifica quais regressores estão disponíveis
    regressors = []
    if 'rainfall_mm' in df.columns:
        regressors.append('rainfall_mm')

    # --- 1. Treinar Modelo de Preço ('y') ---
    m_price = Prophet(
        seasonality_mode='multiplicative', changepoint_prior_scale=0.05,
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False
    )

    df_price_fit = df[['ds', 'y']].copy()
    for reg in regressors:
        m_price.add_regressor(reg)
        df_price_fit[reg] = df[reg]

    m_price.fit(df_price_fit.dropna(subset=['y']))  # Treina apenas com as colunas relevantes

    # Cria datas futuras
    future = m_price.make_future_dataframe(periods=periods)

    # Adiciona dados futuros de regressores SE existirem
    if 'rainfall_mm' in regressors:
        future_dates = pd.to_datetime(future['ds'])
        future['rainfall_mm'] = (
                5 + 15 * np.sin(future_dates.dt.dayofyear / 365 * 2 * np.pi) +
                np.random.normal(0, 3, len(future))
        ).clip(lower=0).round(2)

    forecast_price = m_price.predict(future)

    # --- 2. Treinar Modelo de Volume ('volume') ---
    m_volume = Prophet(
        seasonality_mode='multiplicative', changepoint_prior_scale=0.05,
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False
    )

    df_volume_fit = df[['ds', 'volume']].rename(columns={'volume': 'y'})
    for reg in regressors:
        m_volume.add_regressor(reg)
        df_volume_fit[reg] = df[reg]

    m_volume.fit(df_volume_fit.dropna(subset=['y']))  # Treina apenas com as colunas relevantes
    forecast_volume = m_volume.predict(future)

    # --- 3. Combinar Previsões ---
    forecast = forecast_price.copy()
    forecast['volume_forecast'] = forecast_volume['yhat']
    forecast['volume_lower'] = forecast_volume['yhat_lower']
    forecast['volume_upper'] = forecast_volume['yhat_upper']
    forecast['sales_value_forecast'] = forecast['yhat'] * forecast['volume_forecast']
    forecast['sales_value_lower'] = forecast['yhat_lower'] * forecast['volume_lower']
    forecast['sales_value_upper'] = forecast['yhat_upper'] * forecast['volume_upper']
    return m_price, m_volume, forecast


# --- Função de Tradução e Estilo (ATUALIZADA) ---
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


# --- 3. Interface do App Streamlit (Página Principal) ---
st.set_page_config(layout="wide", page_title="AgroClima - Brasil")

# --- CSS (Simplificado para estabilidade) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    :root { 
        --green-900: #0d5f49; 
        --green-600: #2f8d6f; 
    }
    .stApp { 
        font-family: 'Inter', sans-serif; 
    }
    .stMarkdown h3 { 
        color: var(--green-900); 
        font-weight: 600; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌱 AgroClima - Brasil")
st.markdown("Previsão de Preços e Vendas Agrícolas para o Brasil")

# --- Sidebar Inputs (Sem alterações) ---
with st.sidebar:
    st.markdown(f"<h3>⚙️ Parâmetros do Produtor</h3>", unsafe_allow_html=True)

    crop_choice = st.selectbox(
        "Selecione a Cultura:",
        ["Soja (Soybean)", "Milho (Corn)", "Café Arábica (Arabica Coffee)"]
    )
    region_choice = st.selectbox(
        "Selecione a Região:",
        list(WEATHER_LAT_LON.keys())  # <-- Lê as chaves editadas
    )
    forecast_days = st.slider(
        "Horizonte da Previsão (dias):",
        min_value=30, max_value=730, value=180, step=30
    )
    farm_size = st.radio(
        "Tamanho da Fazenda:",
        ['Pequena (Acessibilidade)', 'Grande (Integração ERP)'],
        index=0
    )

# --- Lógica Principal do Dashboard ---
df_historical, produtividade_padrao = load_and_simulate_data_v5(crop_choice, region_choice)
current_weather = get_current_weather_v5(region_choice)

# 1. Métricas (usando st.metric)
st.markdown(f"## 📍 Dashboard: {crop_choice} em {region_choice}")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("💲 Preço Atual (R$/unid.)", f"R$ {df_historical['y'].iloc[-1]:.2f}")
col2.metric("📦 Volume Atual (ton)", f"{df_historical['volume'].iloc[-1]:.0f}")
col3.metric("🇧🇷 Valor Vendas (R$)", f"R$ {df_historical['sales_value'].iloc[-1]:,.0f}")
col4.metric("🌡️ Temp. Média (Atual)", current_weather["Temperatura Atual"])
col5.metric("💧 Precipitação (24h)", current_weather["Precipitação (24h)"])
col6.metric("☀️ Condição Atual", current_weather["Condição"])

st.markdown("<br>", unsafe_allow_html=True)

# 2. Rodar Modelo
with st.spinner(f"Rodando modelo Prophet... ({forecast_days} dias)"):
    m_price, m_volume, forecast = run_prophet_forecast_v5(df_historical, periods=forecast_days)

# --- 3. LAYOUT EM ABAS (Emojis no lugar de ícones) ---
tab_forecast, tab_risk = st.tabs([
    "📈 Projeção de Tendências",
    "🌦️ Risco Climático & Insights"
])

# --- Aba 1: Gráfico de Projeção ---
with tab_forecast:
    forecast_end = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-1]
    price_change_pct = ((forecast_end['yhat'] - df_historical['y'].iloc[-1]) / df_historical['y'].iloc[-1]) * 100

    col_graph, col_summary = st.columns([3, 1])

    with col_graph:
        # Gráfico principal (Sem alterações)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_historical['ds'], y=df_historical['sales_value'],
            mode='markers+lines', line=dict(color='#1f77b4', width=3),
            marker=dict(color='#1f77b4', size=6, opacity=0.8, line=dict(width=1, color='darkblue')),
            name='Valor de Vendas Real', yaxis='y',
            hovertemplate='<b>Valor de Vendas Real</b><br>Data: %{x|%d/%m/%Y}<br>Valor: R$ %{y:,.0f}<br>Preço: R$ %{customdata[0]:,.2f}<br>Volume: %{customdata[1]:,.0f} ton<extra></extra>',
            customdata=list(zip(df_historical['y'], df_historical['volume']))
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['sales_value_forecast'],
            mode='lines', line=dict(color='#1f77b4', width=2, dash='dash'),
            name='Valor de Vendas Previsto', yaxis='y',
            hovertemplate='<b>Previsão de Valor</b><br>Data: %{x|%d/%m/%Y}<br>Valor: R$ %{y:,.0f}<br>Preço Proj.: R$ %{customdata[0]:,.2f}<br>Volume Proj.: %{customdata[1]:,.0f} ton<extra></extra>',
            customdata=list(zip(forecast['yhat'], forecast['volume_forecast']))
        ))
        fig.add_trace(go.Scatter(
            x=df_historical['ds'], y=df_historical['y'],
            mode='markers+lines', line=dict(color='#ff7f0e', width=3),
            marker=dict(color='#ff7f0e', size=6, opacity=0.8, line=dict(width=1, color='darkorange')),
            name='Preço Real (R$/ton)', yaxis='y2',
            hovertemplate='<b>Preço Real</b><br>Data: %{x|%d/%m/%Y}<br>Preço: R$ %{y:,.2f}<br>Volume: %{customdata[0]:,.0f} ton<br>Valor: R$ %{customdata[1]:,.0f}<extra></extra>',
            customdata=list(zip(df_historical['volume'], df_historical['sales_value']))
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines', line=dict(color='#ff7f0e', width=2, dash='dash'),
            name='Preço Previsto (R$/ton)', yaxis='y2',
            hovertemplate='<b>Previsão de Preço</b><br>Data: %{x|%d/%m/%Y}<br>Preço: R$ %{y:,.2f}<br>Volume Proj.: %{customdata[0]:,.0f} ton<br>Valor Proj.: R$ %{customdata[1]:,.0f}<extra></extra>',
            customdata=list(zip(forecast['volume_forecast'], forecast['sales_value_forecast']))
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
            y=pd.concat([forecast['sales_value_upper'], forecast['sales_value_lower'][::-1]]),
            fill='toself', fillcolor='rgba(31, 119, 180, 0.1)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=False, name='Intervalo de Confiança (Vendas)', yaxis='y'
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(255, 127, 14, 0.1)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=False, name='Intervalo de Confiança (Preço)', yaxis='y2'
        ))
        fig.update_layout(
            title=dict(text=f'🌱 Análise de Vendas e Preços - {crop_choice}', x=0.5,
                       font=dict(size=20, color='#0d5f49')),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=12)),
            height=500, showlegend=True,
            plot_bgcolor='rgba(242, 247, 243, 0.3)', paper_bgcolor='white',
            dragmode='pan', modebar_add=['select2d', 'lasso2d', 'autoScale2d', 'resetScale2d']
        )
        fig.update_xaxes(title_text="Período", showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                         title_font=dict(size=14, color='#0d5f49'))
        fig.update_yaxes(title_text="Valor de Vendas (R$)", tickformat=',.0f', tickprefix="R$ ", showgrid=True,
                         gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_font=dict(size=14, color='#1f77b4'),
                         tickfont=dict(color='#1f77b4'), side='left')
        fig.update_yaxes(title_text="Preço (R$/ton)", tickformat=',.2f', tickprefix="R$ ", showgrid=False,
                         title_font=dict(size=14, color='#ff7f0e'), tickfont=dict(color='#ff7f0e'), side='right',
                         overlaying='y')

        st.plotly_chart(fig, use_container_width=True,
                        config={'displayModeBar': True, 'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                                'displaylogo': False})

    with col_summary:
        st.markdown("### 📊 Resumo da Projeção")
        forecast_end_volume = forecast[['volume_forecast', 'volume_lower', 'volume_upper']].iloc[-1]
        forecast_end_value = forecast[['sales_value_forecast', 'sales_value_lower', 'sales_value_upper']].iloc[-1]
        volume_change_pct = ((forecast_end_volume['volume_forecast'] - df_historical['volume'].iloc[-1]) /
                             df_historical['volume'].iloc[-1]) * 100
        value_change_pct = ((forecast_end_value['sales_value_forecast'] - df_historical['sales_value'].iloc[-1]) /
                            df_historical['sales_value'].iloc[-1]) * 100

        st.markdown(f"""
        **💰 Preço Projetado (R$/ton):** **R$ {forecast_end['yhat']:.2f}**
        - Intervalo: R$ {forecast_end['yhat_lower']:.2f} - R$ {forecast_end['yhat_upper']:.2f}
        - Mudança: {'🟢 +' if price_change_pct > 0 else '🔴 '}{abs(price_change_pct):.1f}%

        **📦 Volume Projetado (ton):** **{forecast_end_volume['volume_forecast']:.0f}**
        - Intervalo: {forecast_end_volume['volume_lower']:.0f} - {forecast_end_volume['volume_upper']:.0f} ton
        - Mudança: {'🟢 +' if volume_change_pct > 0 else '🔴 '}{abs(volume_change_pct):.1f}%

        **💵 Valor de Vendas Projetado:** **R$ {forecast_end_value['sales_value_forecast']:,.0f}**
        - Intervalo: R$ {forecast_end_value['sales_value_lower']:,.0f} - R$ {forecast_end_value['sales_value_upper']:,.0f}
        - Mudança: {'🟢 +' if value_change_pct > 0 else '🔴 '}{abs(value_change_pct):.1f}%
        """)

        if price_change_pct > 5 and volume_change_pct > 0:
            st.success("🚀 **Excelente Oportunidade:** Preço e volume em alta! Considere segurar a colheita.")
        elif price_change_pct > 5 and volume_change_pct < 0:
            st.warning("⚠️ **Oportunidade Mista:** Preço alto, mas volume baixo. Avalie timing de venda.")
        elif price_change_pct < -5 and volume_change_pct > 0:
            st.warning("⚠️ **Situação Crítica:** Preço baixo, mas volume alto. Considere venda antecipada.")
        elif price_change_pct < -5 and volume_change_pct < 0:
            st.error("📉 **Alerta Crítico:** Preço e volume em queda. Analise venda imediata.")
        else:
            st.info("🟡 **Estabilidade:** Foque na otimização de custos e eficiência operacional.")

    st.markdown("---")
    st.markdown("### 🧩 Decompondo a Previsão (Tendência e Sazonalidade)")
    st.info(
        "Este gráfico mostra *por que* o modelo prevê esses preços. Ele separa a 'Tendência' (crescimento de longo prazo) da 'Sazonalidade Anual' (padrões repetitivos, ex: safra/entressafra) e o impacto da 'Chuva' (rainfall_mm).")

    # --- ATUALIZAÇÃO DE TRADUÇÃO ---
    fig_components = plot_components_plotly(m_price, forecast)
    fig_components = melhorar_grafico_componentes(fig_components)  # <-- CHAMA A NOVA FUNÇÃO
    st.plotly_chart(fig_components, use_container_width=True)

# --- Aba 2: Simulação de Risco e Insights (ATUALIZADO) ---
with tab_risk:
    # --- INTRODUÇÃO ADICIONADA ---
    st.info(
        """
        **Por que o Clima é Importante?**
        O clima, especialmente a **precipitação (chuva)**, é um dos fatores mais críticos que afetam a produção agrícola e, consequentemente, os preços.

        A nossa IA (Prophet) usa os dados históricos de chuva como um **regressor**. Isso significa que ela aprende o padrão de como a chuva impactou os preços e os volumes no passado, e usa essa informação para fazer previsões futuras mais precisas.
        """
    )

    st.markdown("### 📈 Projeção de Precipitação (Chuva)")
    df_regressor_hist = df_historical[['ds', 'rainfall_mm']]
    df_regressor_fcst = forecast[['ds', 'rainfall_mm']]

    fig_weather = go.Figure()
    fig_weather.add_trace(go.Scatter(
        x=df_regressor_hist['ds'], y=df_historical['rainfall_mm'],
        mode='lines', name='Precipitação Histórica (mm)',
        line=dict(color='#1f77b4', width=2)
    ))
    fig_weather.add_trace(go.Scatter(
        x=df_regressor_fcst['ds'], y=df_regressor_fcst['rainfall_mm'],
        mode='lines', name='Precipitação Prevista (mm)',
        line=dict(color='#17becf', width=2, dash='dash')
    ))
    fig_weather.update_layout(
        title='Projeção de Precipitação (Fator do Modelo)',  # Título traduzido
        xaxis_title='Período',
        yaxis_title='Precipitação (mm)',
        plot_bgcolor='rgba(242, 247, 243, 0.3)', paper_bgcolor='white',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_weather, use_container_width=True)

    st.markdown("---")
    st.markdown("### ☁️ Simulação de Cenário (O que Aconteceria se?)")
    drought_impact = st.slider(
        "Simular Impacto de Seca/Estiagem (Redução de Precipitação):",
        min_value=0, max_value=50, value=0, step=5, format="%d%%"
    )

    if drought_impact > 0:
        st.warning(
            f"Simulando uma redução de {drought_impact}% na precipitação... Isso historicamente leva a um aumento de 5-15% nos preços, mas uma redução de 10-30% no volume de vendas (rendimento).")
    else:
        st.info("Use o simulador para testar o impacto de secas ou chuvas excessivas no preço do seu produto.")

    st.markdown("---")
    st.markdown("### 🌳 Foco em Sustentabilidade (ESG)")

    if farm_size == 'Pequena (Acessibilidade)':
        st.markdown(
            """
            <div style="background-color: #fff8e1; padding: 15px; border-radius: 8px; border-left: 5px solid #ffc107; margin-top: 15px; color: #795548;">
                <h5 style="color: #795548; margin-top: 0;">💡 Insight para Pequenos Produtores (PRONAF)</h5>
                <p style="font-size: 0.95em; margin: 0;">
                Previsões ligadas a baixas chuvas (secas) devem ser usadas como gatilho para a busca antecipada por 
                financiamento climático, como o <b>Custeio Agropecuário</b> ou renegociação de dívidas. 
                Não espere a perda se concretizar.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 5px solid #4CAF50; margin-top: 15px; color: #1B5E20;">
                <h5 style="color: #1B5E20; margin-top: 0;">🌿 Insight para Grandes Produtores (ESG)</h5>
                <p style="font-size: 0.95em; margin: 0;">
                Use a previsão de precipitação para otimizar o uso de água e fertilizantes. 
                Relatórios que comprovam a redução de insumos com base em dados de previsão 
                aumentam o valor da sua colheita para compradores internacionais focados em <b>metas de ESG</b>.
                </p>
            </div>
            """, unsafe_allow_html=True
        )

# --- Rodapé (Sem alterações) ---
st.markdown("---")
st.markdown(f"""
    <p style="text-align: center; font-size: 0.8em; color: #555;">
    🌱 <b>AgroClima Demo</b> | Tecnologias: Prophet, Plotly, Streamlit.
    </p>
    """, unsafe_allow_html=True
            )