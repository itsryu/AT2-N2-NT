import logging

from src.models import train_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline() -> None:
    logger.info("="*50)
    logger.info("INICIANDO PIPELINE DE TREINAMENTO")
    logger.info("="*50)
    
    try:
        train_model.main()
        logger.info("Pipeline de treinamento finalizado com sucesso.")
    except Exception as e:
        logger.error(f"Ocorreu um erro fatal ao executar o pipeline de treinamento: {e}", exc_info=True)

if __name__ == '__main__':
    run_pipeline()
