# Análise Exploratória de Dados

O dataset utilizado foi o **UrbanTrace: Lifestyle & Pollution Insights**, disponibilizado em formato CSV. A base possui **10.000 registros e 14 colunas**, atendendo ao requisito mínimo de volume definido no trabalho. A descrição original do conjunto informa a presença de valores ausentes, característica que foi confirmada durante a análise exploratória.

## Características Gerais

O problema foi definido como uma tarefa de **classificação multiclasse**, tendo como variável alvo `risk_category`, que representa a categoria de risco de exposição à poluição. As classes presentes são `Low`, `Medium` e `High`.

A distribuição da variável alvo mostrou forte desbalanceamento:

| Classe | Quantidade | Percentual |
| ------ | ---------: | ---------: |
| Medium |      8.587 |     85,87% |
| Low    |      1.198 |     11,98% |
| High   |        215 |      2,15% |

Esse desbalanceamento é um ponto central da análise. Um modelo pode obter alta acurácia apenas ao favorecer a classe majoritária (`Medium`), sem necessariamente aprender bem os padrões das classes minoritárias. Por esse motivo, além da acurácia, foram consideradas métricas como F1-score macro e matriz de confusão.

## Valores Ausentes, Duplicatas e Inconsistências

A base possui valores ausentes em sete variáveis. A coluna com maior proporção de ausência foi `vehicle_ownership`, com 31,64% de valores faltantes. Em seguida aparecem `daily_travel_time`, com 11,24%, e `awareness_level`, com 10,15%.

| Variável           | Valores ausentes | Percentual |
| ------------------ | ---------------: | ---------: |
| vehicle_ownership  |            3.164 |     31,64% |
| daily_travel_time  |            1.124 |     11,24% |
| awareness_level    |            1.015 |     10,15% |
| nearby_industries  |              795 |      7,95% |
| work_location_type |              653 |      6,53% |
| green_space_access |              560 |      5,60% |
| years_in_location  |              520 |      5,20% |

Não foram identificados registros duplicados no dataset. Também não foram encontrados problemas evidentes de tipo nas colunas: as variáveis numéricas foram carregadas como números, as variáveis booleanas como valores lógicos e as variáveis categóricas como texto.

Na etapa de modelagem, os valores ausentes numéricos foram tratados por imputação pela mediana. Essa escolha é robusta contra valores extremos e evita que outliers influenciem excessivamente o preenchimento. Para variáveis categóricas, os valores ausentes foram preenchidos com a categoria `Missing`, preservando a informação de que aquele dado não estava disponível.

## Relações Entre Variáveis

Na análise de correlação entre variáveis numéricas, a coluna `pollution_exposure_score` apresentou as associações mais relevantes com:

| Variável              | Correlação com `pollution_exposure_score` |
| --------------------- | ----------------------------------------: |
| nearby_industries     |                                     0,531 |
| daily_travel_time     |                                     0,301 |
| home_air_quality      |                                    -0,230 |
| years_in_location     |                                     0,164 |
| noise_pollution_level |                                     0,126 |

Esses resultados indicam que a proximidade de indústrias e o tempo diário de deslocamento tendem a estar associados a maior pontuação de exposição à poluição. Já a qualidade do ar em casa possui relação negativa, sugerindo que melhores condições internas estão associadas a menor exposição.

Apesar disso, a coluna `pollution_exposure_score` foi removida das variáveis de entrada dos modelos, pois ela é diretamente relacionada a `risk_category`. Seu uso como preditora poderia causar **vazamento de dados** e gerar resultados artificialmente altos, prejudicando a avaliação real da capacidade preditiva dos modelos.

## Transformações Aplicadas

O pré-processamento foi estruturado com `Pipeline` e `ColumnTransformer`, garantindo que os tratamentos fossem aprendidos apenas no conjunto de treino e aplicados depois ao conjunto de teste. Essa abordagem evita que informações do teste influenciem o treinamento.

As transformações realizadas foram:

- remoção da coluna `pollution_exposure_score` das variáveis preditoras, para evitar vazamento de dados;
- separação entre variáveis numéricas e categóricas;
- preenchimento de valores ausentes categóricos com a categoria `Missing`;
- imputação de valores ausentes numéricos pela mediana;
- padronização das variáveis numéricas com `StandardScaler`;
- codificação das variáveis categóricas com `OneHotEncoder`;
- separação estratificada entre treino e teste, mantendo a proporção das classes.

A padronização foi aplicada porque alguns modelos, como a Regressão Logística, são sensíveis à escala das variáveis. Já a codificação One-Hot foi necessária porque modelos do `scikit-learn` não utilizam diretamente categorias textuais.

## Visualizações Geradas

Para apoiar a análise, foram geradas visualizações no notebook:

- gráfico de barras com a distribuição das classes;
- gráfico de barras com percentual de valores ausentes por variável;
- histogramas das principais variáveis numéricas;
- mapa de calor de correlação entre variáveis numéricas;
- matriz de confusão para avaliar erros de classificação.

Essas visualizações ajudam a justificar as decisões de pré-processamento e evidenciam características importantes do dataset, como o desbalanceamento da variável alvo, a presença de valores ausentes e as relações entre variáveis ambientais e de estilo de vida.
