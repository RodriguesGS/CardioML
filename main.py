from src.pipeline import Pipeline


def main():
    print('=' * 60)
    print('       Trabalho de Inteligência Artificial - 2026')
    print('=' * 60)
    
    pipeline = Pipeline()

    print("\n[1/4] Preparando dados...")
    dataset = pipeline.prepare_data()
    counts = dataset.df["target"].value_counts()

    print("\n[2/4] Treinando modelos...")
    results = pipeline.run()

    print("\n[3/4] Calculando métricas...")
    print(
        f"      Árvore -> Acurácia: {results['tree']['acuracia']:.4f}"
    )
    print(
        f"      Rede MLP -> Acurácia: {results['nn']['acuracia']:.4f}"
    )

    print("\n[4/4] Gerando gráficos e salvando modelos...")
    print("      Arquivos salvos nas pastas models/ e figures/")

    print("\n" + "=" * 60)
    print("Concluído com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
