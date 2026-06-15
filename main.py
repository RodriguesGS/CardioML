from src import pipeline, train_model, evaluate


def main():
    print('==' * 30)
    print('         Trabalho de Inteligência Artificial - 2026')
    print('==' * 30)
    
    print('\n[1/3] Preparando Dados...')
    
    data = pipeline.process_data()
    print(f"      Treino: {len(data['X_train'])} amostras | "
          f"Teste: {len(data['X_test'])} amostras")
    
    print('\n[2/3] Treinando Modelos...')
    print("      Parte 1: Árvore de Decisão")
    print("      Parte 2: Rede Neural MLP")
    
    tree, nn = train_model.train_models(data)
    train_model.save_models(tree, nn, data["scaler"])
    print("      Modelos salvos em models/")
    
    print("\n[3/3] Avaliando e comparando os modelos...")
    evaluate.avaliar_e_comparar(tree, nn, data)
    print("\nGráficos salvos em figures/")
 
    print("\n" + "=" * 56)
    print("  Concluído!")
    print("=" * 56)


if __name__ == "__main__":
    main()
