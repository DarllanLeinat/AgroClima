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
