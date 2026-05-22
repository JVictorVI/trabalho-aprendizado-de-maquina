from __future__ import annotations

from pathlib import Path

import matplotlib

# Usa um backend não interativo para permitir salvar gráficos em PNG sem abrir janelas.
# Isso deixa o script adequado para terminal, VS Code, Jupyter e ambientes sem interface gráfica.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Caminhos principais do projeto. ROOT aponta para a pasta raiz, independentemente
# de onde o script seja chamado.
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "urban_lifestyle_impact_dataset.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"

# Configurações centrais da modelagem.
TARGET = "risk_category"

# Esta coluna é removida da modelagem para evitar vazamento de dados. Ela é uma
# pontuação diretamente relacionada à categoria de risco que queremos prever.
LEAKAGE_COLUMNS = ["pollution_exposure_score"]

# Fixa a aleatoriedade para tornar divisão treino/teste e modelos reprodutíveis.
RANDOM_STATE = 42


def load_data(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Carrega o dataset bruto a partir da pasta data."""
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variáveis explicativas e variável alvo."""
    # X recebe as variáveis preditoras. A variável alvo e a coluna com risco de
    # vazamento são removidas antes do treinamento.
    x = df.drop(columns=[TARGET, *LEAKAGE_COLUMNS])

    # Colunas categóricas e booleanas são convertidas para object antes da divisão.
    # A imputação de valores ausentes continua dentro do Pipeline.
    categorical_features = x.select_dtypes(include=["object", "str", "bool"]).columns.tolist()
    x[categorical_features] = x[categorical_features].astype("object")

    # y é a classe que o modelo deve aprender a prever.
    y = df[TARGET]
    return x, y


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Cria o pré-processador aplicado antes dos modelos."""
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x.select_dtypes(include=["object", "str", "bool"]).columns.tolist()

    # Pipeline para variáveis numéricas:
    # - SimpleImputer(strategy="median") substitui valores ausentes pela mediana.
    # - StandardScaler padroniza as variáveis para média 0 e desvio padrão 1.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Pipeline para variáveis categóricas:
    # - SimpleImputer(strategy="constant") trata ausências como categoria explícita.
    # - OneHotEncoder converte categorias textuais em colunas binárias.
    # - handle_unknown="ignore" evita erro caso apareça uma categoria nova no teste.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # ColumnTransformer permite aplicar tratamentos diferentes para grupos de colunas.
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_models(x_train: pd.DataFrame) -> dict[str, Pipeline]:
    """Define os modelos avaliados, cada um com seu pipeline completo."""
    return {
        "Regressão Logística": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        # Compensa parcialmente o desbalanceamento entre classes.
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        # Evita que o modelo favoreça excessivamente a classe majoritária.
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Random Forest Regularizada": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_split=10,
                        min_samples_leaf=10,
                        max_features="sqrt",
                        # Restringe a complexidade das arvores para reduzir overfitting.
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def save_figure(filename: str) -> Path:
    """Salva a figura atual na pasta outputs/figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_class_distribution(df: pd.DataFrame) -> Path:
    """Gera o gráfico de distribuição da variável alvo."""
    counts = df[TARGET].value_counts()
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index, counts.values, color=colors[: len(counts)])
    ax.set_title("Distribuição das classes de risco")
    ax.set_xlabel("Categoria de risco")
    ax.set_ylabel("Quantidade de registros")
    ax.bar_label(bars, labels=[f"{value:,}".replace(",", ".") for value in counts.values], padding=3)
    ax.set_ylim(0, counts.max() * 1.12)

    return save_figure("01_distribuicao_classes.png")


def plot_missing_values(df: pd.DataFrame) -> Path:
    """Gera o gráfico de percentual de valores ausentes por variável."""
    missing_percent = df.isna().mean().mul(100).sort_values(ascending=True)
    missing_percent = missing_percent[missing_percent > 0]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(missing_percent.index, missing_percent.values, color="#4C78A8")
    ax.set_title("Percentual de valores ausentes por variável")
    ax.set_xlabel("Valores ausentes (%)")
    ax.set_ylabel("Variável")
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in missing_percent.values], padding=3)
    ax.set_xlim(0, max(missing_percent.max() * 1.18, 5))

    return save_figure("02_valores_ausentes.png")


def plot_numeric_histograms(df: pd.DataFrame) -> Path:
    """Gera histogramas das principais variáveis numéricas."""
    columns = [
        "daily_travel_time",
        "nearby_industries",
        "home_air_quality",
        "noise_pollution_level",
        "years_in_location",
        "pollution_exposure_score",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, column in zip(axes.ravel(), columns):
        ax.hist(df[column].dropna(), bins=30, color="#54A24B", edgecolor="white")
        ax.set_title(column)
        ax.set_ylabel("Frequência")

    fig.suptitle("Distribuição das variáveis numéricas", y=1.02)
    return save_figure("03_histogramas_variaveis_numericas.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Gera mapa de calor com correlações entre variáveis numéricas."""
    numeric_columns = df.select_dtypes(include="number").columns
    corr = df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    for row in range(len(corr.columns)):
        for col in range(len(corr.columns)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title("Correlação entre variáveis numéricas")
    return save_figure("04_mapa_correlacao.png")


def plot_model_results(
    results_df: pd.DataFrame,
    y_test: pd.Series,
    best_predictions,
    class_labels: list[str],
    best_model_name: str,
) -> Path:
    """Gera uma figura com métricas comparativas e matriz de confusão."""
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    metric_labels = ["Accuracy", "Macro F1", "Weighted F1"]
    x_positions = range(len(results_df))
    width = 0.22
    model_display_names = {
        "Random Forest Regularizada": "RF\nRegularizada",
        "Regressão Logística": "Regressão\nLogística",
        "Random Forest": "Random\nForest",
    }

    fig, (ax_metrics, ax_matrix) = plt.subplots(1, 2, figsize=(13, 5))

    # Gráfico de barras comparando as métricas dos modelos.
    for offset, metric, label in zip([-width, 0, width], metrics, metric_labels):
        values = results_df[metric].to_numpy()
        positions = [x + offset for x in x_positions]
        bars = ax_metrics.bar(positions, values, width=width, label=label)
        ax_metrics.bar_label(bars, labels=[f"{value:.3f}" for value in values], fontsize=8, padding=2)

    ax_metrics.set_title("Comparação entre modelos")
    ax_metrics.set_ylabel("Valor da métrica")
    ax_metrics.set_ylim(0, 1.05)
    ax_metrics.set_xticks(list(x_positions))
    ax_metrics.set_xticklabels(
        [model_display_names.get(model, model) for model in results_df["model"]],
        rotation=0,
        ha="center",
    )
    ax_metrics.legend(loc="lower right")

    # Matriz de confusão do melhor modelo segundo macro_f1.
    matrix = confusion_matrix(y_test, best_predictions, labels=class_labels)
    image = ax_matrix.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax_matrix, fraction=0.046, pad=0.04)
    ax_matrix.set_title(f"Matriz de confusão - {best_model_name}")
    ax_matrix.set_xlabel("Classe prevista")
    ax_matrix.set_ylabel("Classe real")
    ax_matrix.set_xticks(range(len(class_labels)))
    ax_matrix.set_yticks(range(len(class_labels)))
    ax_matrix.set_xticklabels(class_labels)
    ax_matrix.set_yticklabels(class_labels)

    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax_matrix.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    return save_figure("05_resultados_modelos_e_matriz_confusao.png")


def generate_eda_figures(df: pd.DataFrame) -> list[Path]:
    """Gera as figuras de análise exploratória usadas no relatório técnico."""
    return [
        plot_class_distribution(df),
        plot_missing_values(df),
        plot_numeric_histograms(df),
        plot_correlation_heatmap(df),
    ]


def evaluate_model(
    name: str,
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    predictions,
) -> dict[str, float]:
    """Calcula métricas e imprime um resumo detalhado de avaliação."""
    train_predictions = model.predict(x_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_macro_f1 = f1_score(y_train, train_predictions, average="macro")

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Train macro F1: {train_macro_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, predictions), index=model.classes_, columns=model.classes_))

    return {
        "model": name,
        "train_accuracy": train_accuracy,
        "train_macro_f1": train_macro_f1,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main() -> None:
    """Executa o fluxo completo: EDA, treino, avaliação e geração de figuras."""
    df = load_data()

    # Figuras de EDA são geradas antes da separação treino/teste, pois descrevem a base bruta.
    generated_figures = generate_eda_figures(df)
    x, y = split_features_target(df)

    # A divisão estratificada preserva a proporção das classes em treino e teste.
    # Isso é essencial porque a classe High tem poucos registros.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = build_models(x_train)

    results = []
    predictions_by_model = {}
    for name, model in models.items():
        # Treina o pipeline completo: pré-processamento + algoritmo de classificação.
        model.fit(x_train, y_train)

        # As previsões são feitas apenas no conjunto de teste, simulando dados não vistos.
        predictions = model.predict(x_test)
        predictions_by_model[name] = predictions
        results.append(evaluate_model(name, model, x_train, y_train, x_test, y_test, predictions))

    print("\nResumo comparativo")
    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    print(results_df.to_string(index=False))

    # O melhor modelo é escolhido por macro_f1, métrica mais adequada para classes desbalanceadas.
    best_model_name = results_df.iloc[0]["model"]
    class_labels = sorted(y_test.unique())
    generated_figures.append(
        plot_model_results(
            results_df,
            y_test,
            predictions_by_model[best_model_name],
            class_labels,
            best_model_name,
        )
    )

    print("\nFiguras salvas")
    for path in generated_figures:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
