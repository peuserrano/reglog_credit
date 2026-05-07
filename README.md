# reglog_credit

Previsão de inadimplência de crédito usando regressão logística.

## Descrição

Este projeto constrói um classificador binário para prever se um cliente bancário irá entrar em default no crédito concedido. O modelo usa regressão logística treinada em dados com features de perfil de crédito como score histórico, renda anual, valor de colateral e utilização de cartões.

**Dataset:** [Predicting Co-Branded Credit Card Defaults](https://www.kaggle.com/datasets/thedevastator/predicting-co-branded-credit-card-defaults-in-re) (Kaggle) — transações hipotéticas de um banco fictício.

## Estrutura do Projeto

```
reglog_credit/
├── data/
│   ├── raw/              # CSVs originais (não commitados — ver Setup)
│   └── processed/        # dados intermediários (não commitados)
├── notebooks/
│   └── modelando_default.ipynb
├── src/
│   └── reglog_credit/    # módulos Python reutilizáveis
│       ├── __init__.py
│       ├── data.py
│       └── train.py
├── reports/
│   └── figures/          # gráficos exportados
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── setup.py
└── README.md
```

## Setup

### 1. Clonar o repositório

```bash
git clone https://github.com/<seu-usuario>/reglog_credit.git
cd reglog_credit
```

### 2. Criar e ativar ambiente virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

Para desenvolvimento (linting e testes):

```bash
pip install -r requirements-dev.txt
```

### 4. Baixar o dataset

1. Acesse: https://www.kaggle.com/datasets/thedevastator/predicting-co-branded-credit-card-defaults-in-re
2. Baixe `Training_dataset_Original.csv` e `Data_Dictionary.csv`
3. Coloque ambos em `data/raw/`

### 5. Executar o notebook

```bash
jupyter notebook notebooks/modelando_default.ipynb
```

## Resultados

| Métrica              | Descrição |
|----------------------|-----------|
| Acurácia             | Proporção total de previsões corretas. Enganosa em datasets desbalanceados — um modelo que nunca prevê default já atingiria ~75% de acurácia. |
| Precision (default)  | Dos clientes classificados como default, quantos realmente foram? Alta precision = poucos falsos alarmes (aprovações negadas indevidamente). |
| Recall (default)     | Dos clientes que realmente entraram em default, quantos o modelo identificou? Alto recall = menos defaults passando desapercebidos. |
| F1-Score (default)   | Média harmônica entre Precision e Recall. Útil quando há trade-off entre os dois — penaliza modelos que otimizam só um lado. |
| AUC-ROC              | Área sob a curva ROC. Mede a capacidade de separação entre classes independente do threshold. 1.0 = perfeito, 0.5 = aleatório. |

## Limitações Conhecidas

- Sem análise de threshold ótimo (o padrão 0.5 raramente é o melhor para crédito)
- Sem análise de feature importance por permutação (mais robusta que coeficientes)

## Dependências Principais

| Biblioteca    | Versão  |
|---------------|---------|
| Python        | ≥ 3.10  |
| pandas        | 2.2.2   |
| scikit-learn  | 1.5.0   |
| numpy         | 1.26.4  |
| matplotlib    | 3.9.0   |
| seaborn       | 0.13.2  |
