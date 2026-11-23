## Autores

- **Andre Colabone** - [GitHub](https://github.com/DomColabone)
- **Darllan José Gloor Leinat** - [GitHub](https://github.com/DarllanLeinat)
- **David Correia da Silva** - [GitHub](https://github.com/USERNAME_DAVID](https://github.com/davidsilvaunieduk)

**Orientador:** Prof. Rogério Morandi (UNIMAX - Centro Universitário Max Planck)

# AgroClima Pro

**Sistema de Apoio à Decisão (SAD) baseado em Inteligência Artificial para Previsão Climática e Estimativa de Preços Agrícolas**

Projeto desenvolvido como Trabalho de Conclusão de Curso (TCC) em Ciência da Computação pela **UNIMAX - Centro Universitário Max Planck** (Indaiatuba/SP, 2026).

**Autores:** Andre Colabone, Darllan José Gloor Leinat e David Correia da Silva  
**Orientador:** Prof. Rogério Morandi

---

## Visão Geral

O **AgroClima Pro** é uma plataforma em Python que integra dados climáticos, cotações de mercado e estatísticas produtivas oficiais em um único pipeline ETL, combinando modelagem preditiva de séries temporais (**Prophet**) com uma camada prescritiva de apoio à decisão financeira e operacional no campo.

### Principais Funcionalidades

- **Painel de Previsão e Comercialização (`Painel.py`):** projeção de preços agrícolas (Soja, Milho e Açúcar) para os polos de Mato Grosso (Sinop/Sorriso) e Paraná (Cascavel/Londrina) com bandas de confiança de 80%, identificação de janela ótima de comercialização e orientações de hedge/barter.
- **Clima Histórico e Simulador de Choques (Aba 2):** acompanhamento histórico de precipitação e temperatura com simulador heurístico de sensibilidade a choques climáticos.
- **Inteligência Agroclimática Operacional (Aba 3):** janela de pulverização hora a hora (72 horas) baseada em critérios agronômicos de deriva, inversão térmica e evaporação (Embrapa), balanço hídrico do solo (CAD/Kc conforme FAO-56) e radar de alertas em tempo real.
- **Calculadora Avançada do Talhão (`pages/2_Calculadora_Avançada.py`):** análise financeira em granularidade de talhão (receita bruta, custo total operacional, lucro líquido e margem), simulador de *Stress Test* (quebra de produtividade vs. elasticidade de preço e ponto de ruptura), **Score de Resiliência Climática (0 a 100 / Graus A, B e C)** e analisador de planilhas `.csv`/`.xlsx` sob demanda.
- **Geocodificação Híbrida com Triplo Fallback (`data_fetchers/geocoding_util.py`):** resolução de municípios brasileiros em cascata (base offline com 53 polos agrícolas -> Open-Meteo Geocoding API -> OpenStreetMap Nominatim).

---

## Fontes de Dados Integradas

| Fonte | Variáveis | Papel no Sistema |
| :--- | :--- | :--- |
| **Yahoo Finance (`yfinance`)** | Futuros de Soja (`ZS=F`), Milho (`ZC=F`), Açúcar (`SB=F`) e câmbio `BRL=X` com ajuste fixo de *basis* regional (MT e PR) | ETL -> SQLite -> Treinamento e inferência no `Painel.py` |
| **Open-Meteo Archive API** | Temperatura máxima/mínima e precipitação diária (2005–2026) | ETL -> SQLite -> Regressor climático (`rainfall_mm`) e visualização |
| **Open-Meteo Forecast API** | Previsão horária (72h) e diária (28d, incluindo ET0 FAO) | Módulo operacional em tempo real (pulverização, CAD e alertas) |
| **IBGE SIDRA API (Tabela 1612 - PAM)** | Produção anual (t) e rendimento médio (kg/ha) de Soja e Milho (MT e PR) | Indicadores informativos macroeconômicos no painel |
| **CSV Local (`data/cepea_precos.csv`)** | Séries históricas locais (2015–2025, 3.961 registros) | Contingência (*fallback*) caso o Yahoo Finance fique indisponível |

---

## Arquitetura do Repositório

```text
AgroClima/
├── Painel.py                          # Dashboard principal (Streamlit + Plotly + Prophet)
├── data_pipeline.py                   # Pipeline ETL e persistência no SQLite
├── calculadora_core.py                # Equações financeiras e de stress test do talhão
├── auditoria_tcc.py                   # Script isolado de validação walk-forward e geração de gráficos
├── data_fetchers/
│   ├── commodity_fetcher.py           # Ingestão Yahoo Finance + fallback CSV
│   ├── weather_fetcher.py             # Ingestão Open-Meteo Archive (FlatBuffers + cache)
│   ├── ibge_fetcher.py                # Ingestão IBGE SIDRA (PAM)
│   ├── zeus_intelligence.py           # Regras operacionais (pulverização 72h, CAD/Kc, alertas)
│   └── geocoding_util.py              # Geocodificação em triplo fallback
├── pages/
│   └── 2_Calculadora_Avançada.py      # Calculadora por talhão, Stress Test, Score e Analisador CSV
├── tests/
│   └── test_calculadora.py            # Testes unitários automatizados (pytest)
├── data/
│   ├── agriforecast_unified_db.sqlite # Base consolidada (7.964 dias únicos por UF, 2005–2026)
│   └── cepea_precos.csv               # Base local de contingência
└── graficos_tcc/                      # Gráficos gerados na validação experimental
```

## Validação Experimental e Resultados

Para avaliar a generalização *out-of-sample* sem vazamento de dados, o script `auditoria_tcc.py` executa validação **walk-forward com janela expansiva** sobre a janela recente de 5 anos (`22/10/2021` a `21/10/2026`), em **4 cortes não sobrepostos de 90 dias** (360 dias *out-of-sample*).

### Média consolidada (4 cortes de 90 dias)

| Cultura / UF | Baseline Ingênuo (MAE / MAPE) | Prophet Univariado (MAE / MAPE) | Prophet + Chuva (MAE / MAPE) |
| :--- | :---: | :---: | :---: |
| **Soja - MT** | R$ 4,74 / 3,99% | R$ 6,99 / 5,76% | R$ 6,98 / 5,75% |
| **Soja - PR** | R$ 5,55 / 3,99% | R$ 8,18 / 5,76% | R$ 8,19 / 5,77% |
| **Milho - MT** | R$ 1,77 / 4,00% | R$ 4,01 / 8,99% | R$ 4,01 / 8,98% |
| **Açúcar - MT** | R$ 7,50 / 7,71% | R$ 13,10 / 13,88% | R$ 13,58 / 14,43% |

- **Comportamento por regime:** na Soja (MT), o Prophet manteve MAPE entre **2,70% e 4,21%** nos 3 primeiros cortes (superando o baseline no Fold 2: `3,73%` vs. `5,23%`), sendo impactado no Fold 4 por um choque exógeno de alta na cotação (pico de erro de 19,8% em 27/09/2026).
- **Efeito climático local:** a precipitação diária municipal apresentou impacto marginal (`-0,01 p.p.` na Soja/MT), evidenciando que o clima afeta cotações internacionais de forma acumulada e macro-regional.
- **Qualidade e usabilidade:** 5/5 testes unitários aprovados em `pytest` (`calculadora_core.py`) e avaliação preliminar positiva com 3 profissionais do agronegócio (nota 5/5 em usabilidade e na Calculadora do Talhão).

---

## Como Executar Localmente

### 1. Instalar dependências

```bash
git clone [https://github.com/DarllanLeinat/AgroClima.git](https://github.com/DarllanLeinat/AgroClima.git)
cd AgroClima
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
