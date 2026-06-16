from src.pipeline import Pipeline
from src.config import stp_log

log = stp_log()

def main():
    print('=' * 60)
    print('       Trabalho de Inteligência Artificial - 2026')
    print('=' * 60)
    
    log.info('Iniciando pipeline\n')
    pipeline = Pipeline()

    log.info('[1/4] Preparando dados...')
    dataset = pipeline.prepare_data()
    log.info('Dados prontos')
    counts = dataset.df['target'].value_counts()
    log.info(f'{counts[1]} com doença / {counts[0]} sem doença\n')

    log.info('[2/4] Treinando modelos...')
    pipeline.train_models()
    
    log.info('[3/4] Calculando métricas...')
    results = pipeline.evaluate()
    log.info(f'Árvore   - Acurácia: {(results['tree']['acuracia']) * 100:.2f}%')
    log.info(f'Rede MLP - Acurácia: {(results['nn']['acuracia']) * 100:.2f}%\n')

    log.info('[4/4] Gerando gráficos e salvando modelos...')
    pipeline.generate_graphs()
    log.info('Arquivos salvos nas pastas models/ e figures/\n')

    log.info('Pipeline finalizada com sucesso!!')

if __name__ == '__main__':
    main()
