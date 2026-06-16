import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

from tqdm import tqdm
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from .config import FIGURES_DIR

matplotlib.use('Agg')

class VisualizerGraphs:
    
    def __init__(self, dataset, tree, nn, metrics_tree, metrics_nn):
        
        self.dataset = dataset
        self.tree = tree
        self.nn = nn
        
        self.metrics_tree = metrics_tree
        self.metrics_nn = metrics_nn
        
        
    def distribuition(self):
        
        df = self.dataset.df
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.patch.set_facecolor("#fff")

        counts = df["target"].value_counts()
        
        axes[0].bar(
            ["Sem doença (0)", "Com doença (1)"],
            [counts[0], counts[1]],
            color=['#2563EB', '#DC2626'],
            width=0.5, edgecolor="#fff", linewidth=1.5
        )
        
        axes[0].set_title("Distribuição da Variável Alvo", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("Quantidade de Pacientes")
        
        for i, v in enumerate([counts[0], counts[1]]):
            axes[0].text(i, v + 2, str(v), ha="center", fontweight="bold")

        axes[1].hist(
            df[df["target"] == 0]["age"], 
            bins=20, 
            alpha=0.7,
            color='#2563EB', 
            label="Sem doença"
        )
        
        axes[1].hist(
            df[df["target"] == 1]["age"], 
            bins=20,
            alpha=0.7,
            color='#DC2626', 
            label="Com doença")
        
        axes[1].set_title("Distribuição de Idade por Diagnóstico", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Idade")
        axes[1].set_ylabel("Frequência")
        axes[1].legend()

        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "grafico_distribuicao.png", dpi=150, bbox_inches="tight", facecolor="#fff")
        
    
    def correlation(self):     
        
        fig, ax = plt.subplots(figsize=(11, 8))
        fig.patch.set_facecolor("#fff")

        corr = self.dataset.df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.8, center=0, square=True,
                    linewidths=0.5, annot=True, fmt=".2f", annot_kws={"size": 8}, ax=ax)
        ax.set_title("Matriz de Correlação", fontsize=13, fontweight="bold", pad=15)

        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "grafico_correlacao.png", dpi=150, bbox_inches="tight", facecolor="#fff")
            
    def decision_tree(self):
        
        fig, ax = plt.subplots(figsize=(22, 10))
        fig.patch.set_facecolor("#fff")
         
        class_names = ['Sem doença (0)', 'Com doença (1)']

        plot_tree(self.tree, feature_names=self.dataset.X_train.columns, class_names=class_names,
                filled=True, rounded=True, fontsize=8, ax=ax)
        ax.set_title("Árvore de Decisão", fontsize=14, fontweight="bold", pad=15)

        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "grafico_arvore.png", dpi=120, bbox_inches="tight", facecolor="#fff")

    def confusion_tree(self):
        
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#fff")
        
        class_names = ['Sem doença (0)', 'Com doença (1)']
 
        cm = confusion_matrix(self.dataset.y_test, self.metrics_tree["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor="#fff", annot_kws={"size": 14, "weight": "bold"})
        ax.set_title("Matriz de Confusão — Árvore de Decisão", fontsize=12, fontweight="bold")
        ax.set_ylabel("Real")
        ax.set_xlabel("Predição")
 
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "matriz_arvore.png", dpi=150, bbox_inches="tight", facecolor="#fff")

    def confusion_nn(self):
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.patch.set_facecolor("#fff")
 
        epocas = range(1, len(self.nn.loss_curve_) + 1)
        axes[0].plot(epocas, self.nn.loss_curve_, color='#2563EB',
                     linewidth=2, label="Treino (Loss)")
        if hasattr(self.nn, "validation_scores_") and self.nn.validation_scores_:
            axes[0].plot(epocas, self.nn.validation_scores_, color='#DC2626',
                         linewidth=2, linestyle="--", label="Validação (Acurácia)")
        axes[0].set_title("Curva de Perda — Rede Neural", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Épocas")
        axes[0].set_ylabel("Loss / Acurácia")
        axes[0].legend()
        
        class_names = ['Sem doença (0)', 'Com doença (1)']
 
        cm = confusion_matrix(self.dataset.y_test, self.metrics_nn["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor="#fff", annot_kws={"size": 14, "weight": "bold"})
        axes[1].set_title("Matriz de Confusão — Rede Neural", fontsize=13, fontweight="bold")
        axes[1].set_ylabel("Real")
        axes[1].set_xlabel("Predição")
 
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "grafico_redeNeural.png", dpi=150, bbox_inches="tight", facecolor="#fff")

    def comparison(self):
        
        names = ["Acurácia", "Precisão", "Recall", "F1-Score"]
        keys = ["acuracia", "precisao", "recall", "f1"]
        score_tree = [self.metrics_tree[k] for k in keys]
        score_nn = [self.metrics_nn[k] for k in keys]
 
        x = np.arange(len(names))
 
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("white")
 
        bar1 = ax.bar(x - 0.33 / 2, score_tree, 0.33, label="Árvore de Decisão",
                         color='#2563EB', edgecolor="white", linewidth=1.5)
        bar2 = ax.bar(x + 0.33 / 2, score_nn, 0.33, label="Rede Neural (MLP)",
                         color='#DC2626', edgecolor="white", linewidth=1.5)
 
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Comparação de Desempenho — Árvore de Decisão vs. Rede Neural",
                     fontsize=13, fontweight="bold", pad=12)
        ax.legend(fontsize=10)
 
        for bar in (*bar1, *bar2):
            cor = '#2563EB' if bar in bar1 else '#DC2626'
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=cor)
 
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "comparacao.png", dpi=150, bbox_inches="tight", facecolor="#fff")
        
    def generate_all(self):

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
 
        graphs = [
            self.distribuition,
            self.correlation,
            self.decision_tree,
            self.confusion_tree,
            self.confusion_nn,
            self.comparison,
        ]
 
        bar = tqdm(graphs, desc="Gerando gráficos", ncols=70,
                     bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}")
        for generate in bar:
            try:
                generate()
            except Exception as err:
                print(f"\n    Erro ao gerar {generate.__name__}: {err}")
                