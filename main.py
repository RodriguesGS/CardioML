from src import pipeline, train_model, evaluate


def main():
    print('==' * 30)
    print('      Trabalho de Inteligência Artificial - Pipeline')
    print('==' * 30)
    
    print('\n[1/3] Preparando Dados...')
    
    data = pipeline.process_data()
    print(
        f'''      Treino: {len(data['X_train'])} amostras 
      Teste: {len(data['X_test'])} amostras
        ''')
    
    print('\n[2/3] Treinando Modelos...')
    print('      - Árvore de Decisão')
    print('      - Rede Neural MLP')
    
    tree, nn = train_model.train_models(data)
    train_model.save_models(tree, nn, data['scaler'])
    print('\n      > Os modelos foram salvos em models/')
    
    print('\n[3/3] Avaliando e comparando os modelos...')
    evaluate.avaliar_e_comparar(tree, nn, data)
    print('\nGráficos salvos em figures/')
    
    print('==' * 20)
    print('      Finalizado')
    print('==' * 20)


if __name__ == "__main__":
    main()
