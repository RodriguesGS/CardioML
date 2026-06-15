import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
from sklearn.tree import plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"

AZUL = "#2563EB"
VERMELHO = "#DC2626"
CLASSES = ["Sem Doença", "Com Doença"]


def _configurar_estilo():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.facecolor": "#FAFAFA",
    })


def calcular_metricas(modelo, X_test, y_test):
    """Calcula acurácia, precisão, recall e F1-score de um modelo."""
    y_pred = modelo.predict(X_test)
    return {
        "acuracia": accuracy_score(y_test, y_pred),
        "precisao": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "y_pred": y_pred,
    }


def grafico_distribuicao(df):
    """Figura 1 — Distribuição da variável alvo e de idade por diagnóstico."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("white")

    counts = df["target"].value_counts()
    axes[0].bar(["Sem doença (0)", "Com doença (1)"], [counts[0], counts[1]],
                color=[AZUL, VERMELHO], width=0.5, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Distribuição da Variável Alvo", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Quantidade de Pacientes")
    for i, v in enumerate([counts[0], counts[1]]):
        axes[0].text(i, v + 2, str(v), ha="center", fontweight="bold")

    axes[1].hist(df[df["target"] == 0]["age"], bins=20, alpha=0.7,
                 color=AZUL, label="Sem doença", edgecolor="white")
    axes[1].hist(df[df["target"] == 1]["age"], bins=20, alpha=0.7,
                 color=VERMELHO, label="Com doença", edgecolor="white")
    axes[1].set_title("Distribuição de Idade por Diagnóstico", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Idade")
    axes[1].set_ylabel("Frequência")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_distribuicao.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grafico_correlacao(df):
    """Figura 2 — Matriz de correlação entre as features."""
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("white")

    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.8, center=0, square=True,
                linewidths=0.5, annot=True, fmt=".2f", annot_kws={"size": 8}, ax=ax)
    ax.set_title("Matriz de Correlação", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_correlacao.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grafico_arvore(arvore, feature_names):
    """Figura 3 — Visualização da Árvore de Decisão."""
    fig, ax = plt.subplots(figsize=(22, 10))
    fig.patch.set_facecolor("white")

    plot_tree(arvore, feature_names=feature_names, class_names=CLASSES,
              filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Árvore de Decisão — Diagnóstico de Doença Cardíaca",
                 fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_arvore.png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grafico_importancia(arvore, feature_names):
    """Figura 4 — Importância das features (critério Gini)."""
    importancias = pd.Series(arvore.feature_importances_, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")

    cores = [VERMELHO if v > 0.08 else AZUL for v in importancias.values]
    importancias.plot.barh(ax=ax, color=cores, edgecolor="white")
    ax.set_title("Importância das Features — Árvore de Decisão", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importância (Critério Gini)")
    for i, v in enumerate(importancias.values):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_importancia.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grafico_matriz_confusao_arvore(y_test, y_pred):
    """Figura 5 — Matriz de confusão da Árvore de Decisão."""
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("white")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax,
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.5, linecolor="white", annot_kws={"size": 14, "weight": "bold"})
    ax.set_title("Matriz de Confusão — Árvore de Decisão", fontsize=12, fontweight="bold")
    ax.set_ylabel("Real")
    ax.set_xlabel("Predito")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_matriz_arvore.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grafico_rede_neural(rede_neural, y_test, y_pred):
    """Figura 6 — Curva de perda (loss) e matriz de confusão da Rede Neural."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("white")

    epocas = range(1, len(rede_neural.loss_curve_) + 1)
    axes[0].plot(epocas, rede_neural.loss_curve_, color=AZUL, linewidth=2, label="Treino (Loss)")
    if hasattr(rede_neural, "validation_scores_") and rede_neural.validation_scores_:
        axes[0].plot(epocas, rede_neural.validation_scores_, color=VERMELHO, linewidth=2,
                     linestyle="--", label="Validação (Accuracy)")
    axes[0].set_title("Curva de Perda — Rede Neural", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Épocas")
    axes[0].set_ylabel("Loss / Acurácia")
    axes[0].legend()

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.5, linecolor="white", annot_kws={"size": 14, "weight": "bold"})
    axes[1].set_title("Matriz de Confusão — Rede Neural", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Real")
    axes[1].set_xlabel("Predito")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_rede_neural.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grafico_comparacao(metricas_arvore, metricas_rede):
    """Figura 7 — Comparação gráfica das métricas entre os dois modelos."""
    nomes = ["Acurácia", "Precisão", "Recall", "F1-Score"]
    chaves = ["acuracia", "precisao", "recall", "f1"]
    scores_arvore = [metricas_arvore[k] for k in chaves]
    scores_rede = [metricas_rede[k] for k in chaves]

    x = np.arange(len(nomes))
    largura = 0.33

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    barras1 = ax.bar(x - largura / 2, scores_arvore, largura, label="Árvore de Decisão",
                     color=AZUL, edgecolor="white", linewidth=1.5)
    barras2 = ax.bar(x + largura / 2, scores_rede, largura, label="Rede Neural (MLP)",
                     color=VERMELHO, edgecolor="white", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(nomes, fontsize=11)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Comparação de Desempenho — Árvore de Decisão vs. Rede Neural",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)

    for barra in barras1:
        ax.text(barra.get_x() + barra.get_width() / 2.0, barra.get_height() + 0.005,
                f"{barra.get_height():.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=AZUL)
    for barra in barras2:
        ax.text(barra.get_x() + barra.get_width() / 2.0, barra.get_height() + 0.005,
                f"{barra.get_height():.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=VERMELHO)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_comparacao.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def imprimir_comparacao(metricas_arvore, metricas_rede):
    """Imprime a tabela comparativa final no terminal."""
    print("\n" + "=" * 56)
    print(f"{'Métrica':<12}{'Árvore Decisão':>18}{'Rede Neural':>18}")
    print("-" * 56)
    linhas = [("Acurácia", "acuracia"), ("Precisão", "precisao"),
              ("Recall", "recall"), ("F1-Score", "f1")]
    for nome, chave in linhas:
        marca_a = " *" if metricas_arvore[chave] >= metricas_rede[chave] else "  "
        marca_r = " *" if metricas_rede[chave] > metricas_arvore[chave] else "  "
        print(f"{nome:<12}{metricas_arvore[chave]:>16.4f}{marca_a}{metricas_rede[chave]:>16.4f}{marca_r}")
    print("=" * 56)
    print("(* = melhor resultado na métrica)")


def avaliar_e_comparar(arvore, rede_neural, dados):
    """
    Executa toda a avaliação: calcula métricas, gera os gráficos e imprime
    a comparação final.

    Retorna as métricas de ambos os modelos.
    """
    _configurar_estilo()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = list(dados["X_train"].columns)

    # Métricas
    metricas_arvore = calcular_metricas(arvore, dados["X_test"], dados["y_test"])
    metricas_rede = calcular_metricas(rede_neural, dados["X_test_scaled"], dados["y_test"])

    # Gráficos exploratórios
    grafico_distribuicao(dados["df"])
    grafico_correlacao(dados["df"])

    # Gráficos da Árvore
    grafico_arvore(arvore, feature_names)
    grafico_importancia(arvore, feature_names)
    grafico_matriz_confusao_arvore(dados["y_test"], metricas_arvore["y_pred"])

    # Gráficos da Rede Neural
    grafico_rede_neural(rede_neural, dados["y_test"], metricas_rede["y_pred"])

    # Comparação
    grafico_comparacao(metricas_arvore, metricas_rede)
    imprimir_comparacao(metricas_arvore, metricas_rede)

    return metricas_arvore, metricas_rede
