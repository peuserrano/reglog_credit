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

| Métrica           | Valor |
|-------------------|-------|
| Acurácia          | —     |
| Precision (default) | —   |
| Recall (default)  | —     |
| F1-Score (default) | —    |
| AUC-ROC           | —     |

*Preencher após execução do notebook.*

## Limitações Conhecidas

- Sem cross-validation ou tuning de hiperparâmetros
- Desbalanceamento de classes não tratado com resampling (SMOTE/undersampling)
- Apenas regressão logística avaliada (sem comparação com outros modelos)

## Dependências Principais

| Biblioteca    | Versão  |
|---------------|---------|
| Python        | ≥ 3.10  |
| pandas        | 2.2.2   |
| scikit-learn  | 1.5.0   |
| numpy         | 1.26.4  |
| matplotlib    | 3.9.0   |
| seaborn       | 0.13.2  |
