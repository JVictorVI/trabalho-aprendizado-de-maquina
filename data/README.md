# Dataset - UrbanTrace: Lifestyle & Pollution Insights

Esta pasta contém os dados brutos utilizados no Trabalho 2 de Aprendizado de Máquina.

## Arquivos

- `urban_lifestyle_impact_dataset.csv`: dataset principal do projeto.

## Visão geral

O dataset **UrbanTrace: Lifestyle & Pollution Insights** reúne informações sintéticas sobre estilo de vida, deslocamento e ambiente urbano, com o objetivo de analisar risco de exposição à poluição.

Características gerais:

- **Formato:** CSV
- **Quantidade de registros:** 10.000
- **Quantidade de colunas:** 14
- **Contém valores ausentes:** Sim
- **Tipo de problema usado no projeto:** Classificação multiclasse
- **Variável alvo escolhida:** `risk_category`

O dataset também permite uma tarefa de regressão usando `pollution_exposure_score`, mas neste projeto foi escolhida a tarefa de classificação.

## Variável alvo

| Coluna          | Tipo       | Descrição                                                                               |
| --------------- | ---------- | --------------------------------------------------------------------------------------- |
| `risk_category` | Categórica | Categoria de risco de exposição à poluição. Possui as classes `Low`, `Medium` e `High`. |

Distribuição observada da variável alvo:

| Classe   | Quantidade | Percentual |
| -------- | ---------: | ---------: |
| `Medium` |      8.587 |     85,87% |
| `Low`    |      1.198 |     11,98% |
| `High`   |        215 |      2,15% |

Essa distribuição mostra que o dataset é desbalanceado, com predominância da classe `Medium`.

## Descrição das colunas

| Coluna                     | Tipo       | Descrição                                                                                       |
| -------------------------- | ---------- | ----------------------------------------------------------------------------------------------- |
| `daily_travel_time`        | Numérica   | Tempo diário de deslocamento. Pode indicar maior exposição a trânsito, poluição urbana e ruído. |
| `vehicle_ownership`        | Categórica | Tipo de posse ou uso de veículo, como transporte público, veículo próprio ou duas rodas.        |
| `location_type`            | Categórica | Tipo de localização em que a pessoa vive, como área urbana, semiurbana ou rural.                |
| `nearby_industries`        | Numérica   | Indicador da quantidade ou proximidade de indústrias próximas ao indivíduo.                     |
| `green_space_access`       | Categórica | Nível de acesso a áreas verdes, como baixo, moderado ou alto.                                   |
| `home_air_quality`         | Numérica   | Medida da qualidade do ar no ambiente residencial.                                              |
| `work_location_type`       | Categórica | Tipo de local de trabalho, como escritório, fábrica, remoto ou outro tipo de ambiente.          |
| `smoker_in_household`      | Booleana   | Indica se existe fumante no domicílio.                                                          |
| `noise_pollution_level`    | Numérica   | Nível de poluição sonora associado ao ambiente.                                                 |
| `use_of_air_purifiers`     | Booleana   | Indica se há uso de purificadores de ar.                                                        |
| `awareness_level`          | Categórica | Nível de consciência ou conhecimento sobre questões ambientais e poluição.                      |
| `years_in_location`        | Numérica   | Número de anos vivendo ou permanecendo na localização atual.                                    |
| `pollution_exposure_score` | Numérica   | Pontuação contínua de exposição à poluição. Foi usada apenas na análise exploratória.           |
| `risk_category`            | Categórica | Classe final de risco de exposição à poluição. É a variável alvo da classificação.              |

## Valores ausentes

Foram encontrados valores ausentes nas seguintes colunas:

| Coluna               | Valores ausentes | Percentual |
| -------------------- | ---------------: | ---------: |
| `vehicle_ownership`  |            3.164 |     31,64% |
| `daily_travel_time`  |            1.124 |     11,24% |
| `awareness_level`    |            1.015 |     10,15% |
| `nearby_industries`  |              795 |      7,95% |
| `work_location_type` |              653 |      6,53% |
| `green_space_access` |              560 |      5,60% |
| `years_in_location`  |              520 |      5,20% |

No projeto, os valores ausentes foram tratados da seguinte forma:

- variáveis numéricas: imputação pela mediana;
- variáveis categóricas: preenchimento com a categoria `Missing`.

## Observações importantes para modelagem

A coluna `pollution_exposure_score` não foi utilizada como variável preditora no modelo de classificação deste projeto, pois ela é diretamente relacionada à variável alvo `risk_category`.

Usar essa coluna no treinamento poderia causar **vazamento de dados**, fazendo o modelo receber uma informação muito próxima da resposta esperada e gerando métricas artificialmente altas.

Por isso, no código do projeto, as variáveis removidas antes do treinamento são:

```python
TARGET = "risk_category"
LEAKAGE_COLUMNS = ["pollution_exposure_score"]
```

## Fonte

Dataset: **UrbanTrace: Lifestyle & Pollution Insights**  
Fonte: Kaggle  
Uso no projeto: classificação da categoria `risk_category`.
