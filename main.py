from src import pipeline



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


if __name__ == "__main__":
    main()
