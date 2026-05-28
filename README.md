# Trabalho 2 - Aprendizado de Máquina

Projeto de **classificação multiclasse** usando o dataset **UrbanTrace: Lifestyle & Pollution Insights**.

## Equipe

- Amanda Evellin de Sousa Viana (2315774)
- João Victor da Silva Ferreira (2314387)
- Paulo Marconi Araújo Tomaz da Silva (2310435)
- Pedro Enrique Jordão Ramos Gama e Silva (2315773)
- Rogério Bruno de Almeida Júnior (2316922)

## Dataset

O projeto utiliza o dataset **UrbanTrace: Lifestyle & Pollution Insights**, disponível publicamente no Kaggle:  
https://www.kaggle.com/datasets/umuttuygurr/city-lifestyle-segmentation-dataset

O conjunto de dados foi desenvolvido para simular cenários urbanos modernos e analisar como fatores relacionados ao estilo de vida, mobilidade urbana e condições ambientais influenciam a exposição à poluição. O dataset combina informações demográficas, hábitos cotidianos e características ambientais para permitir estudos de segmentação, análise preditiva e classificação de risco.

A base contém **10.000 registros** e **14 atributos**, incluindo variáveis numéricas, categóricas e booleanas. Entre as informações presentes estão:

- tempo diário de deslocamento;
- posse de veículo;
- frequência de uso de transporte público;
- proximidade de áreas industriais;
- acesso a áreas verdes;
- qualidade do ar dentro de casa;
- nível de ruído urbano;
- tipo de região residencial;
- indicadores de exposição à poluição.

O alvo do projeto é a coluna `risk_category`, que classifica o nível de risco de exposição à poluição em três categorias:

- `Low`
- `Medium`
- `High`

O dataset apresenta características relevantes para problemas reais de aprendizado de máquina, como:

- presença de valores ausentes;
- desbalanceamento entre classes;
- mistura de variáveis numéricas e categóricas;
- necessidade de pré-processamento e transformação de atributos.

Essas características tornam a base adequada para experimentos de classificação supervisionada, comparação entre modelos e avaliação de técnicas de pré-processamento e regularização. Além disso, o tema do dataset possui relevância prática por abordar impactos ambientais e qualidade de vida em contextos urbanos.

## Arquivos

- `data/urban_lifestyle_impact_dataset.csv`: dataset usado no projeto.
- `data/README.md`: documentação das colunas e características do dataset.
- `notebooks/`: notebooks de análise e modelagem.
- `notebooks/Análise Exploratória de Dados.md`: Documento de EDA, contendo comentários sobre os processos de pré-processamento, modelagem e avaliação.
- `notebooks/Classificação - UrbanTrace.ipynb`: notebook principal do projeto.
- `src/train_classification.py`: script executável com pré-processamento, treino e avaliação.
- `requirements.txt`: dependências necessárias para executar o projeto.

## Estrutura do projeto

```text
trabalho-aprendizado-de-maquina/
├── data/
│   ├── README.md
│   └── urban_lifestyle_impact_dataset.csv
├── notebooks/
|   |   Análise Exploratória de Dados.md
│   └── Classificação - UrbanTrace.ipynb
├── src/
│   └── train_classification.py
├── README.md
└── requirements.txt
```

## Como executar

Crie um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Se aparecer erro do NumPy citando arquivos `cp312` em um ambiente Python 3.13, recrie o ambiente virtual do zero:

```powershell
Remove-Item -Recurse -Force .venv
& "C:\Users\jvict\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Execute o treinamento pelo script:

```bash
python src/train_classification.py
```

Ao executar o script, os modelos são treinados e os gráficos são gerados automaticamente em `outputs/figures/`.

Ou abra o notebook principal:

```bash
jupyter notebook "notebooks/Classificação - UrbanTrace.ipynb"
```

## Metodologia

O projeto usa `risk_category` como variável alvo. A coluna `pollution_exposure_score` foi removida das variáveis preditoras porque está diretamente relacionada ao alvo e poderia causar **vazamento de dados**. Se essa coluna fosse usada no treinamento, o modelo poderia aprender uma aproximação direta da resposta, gerando métricas artificialmente altas.

O pré-processamento foi estruturado com `Pipeline` e `ColumnTransformer`, seguindo boas práticas de aprendizado de máquina:

- valores numéricos ausentes: imputação pela mediana;
- valores categóricos ausentes: preenchimento com a categoria `Missing`;
- variáveis categóricas: codificação com `OneHotEncoder`;
- variáveis numéricas: padronização com `StandardScaler`;
- divisão treino/teste: separação estratificada para preservar a proporção das classes.

Foram comparados três modelos:

- Regressão Logística com `class_weight="balanced"`;
- Random Forest com `class_weight="balanced"`;
- Random Forest Regularizada com `class_weight="balanced"`, `max_depth=12`, `min_samples_split=10`, `min_samples_leaf=10` e `max_features="sqrt"`.

Como as classes são desbalanceadas, a avaliação considera não apenas `accuracy`, mas também `macro F1`, `weighted F1`, relatório de classificação e matriz de confusão.

## Resultados iniciais

Com divisão estratificada de 80% para treino e 20% para teste:

| Modelo                     | Accuracy | Macro F1 | Weighted F1 |
| -------------------------- | -------: | -------: | ----------: |
| Random Forest Regularizada |   0.9120 |   0.7652 |      0.9156 |
| Regressão Logística        |   0.8820 |   0.7190 |      0.8960 |
| Random Forest              |   0.9080 |   0.5663 |      0.8879 |

A Random Forest sem restrições apresentou desempenho perfeito no treino e queda forte no `macro F1` de teste, indicando overfitting. A versão regularizada reduziu esse problema e passou a apresentar o melhor `macro F1`, indicando melhor equilíbrio entre as classes. Como o dataset é desbalanceado, especialmente pela baixa quantidade de exemplos da classe `High`, o `macro F1` deve ter peso maior na discussão dos resultados.
