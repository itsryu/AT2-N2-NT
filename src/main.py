import logging

from src.models import train_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline() -> None:
    try:
        train_model.main()
    except Exception as e:
        logger.error(f"Ocorreu um erro fatal ao executar o pipeline de treinamento: {e}", exc_info=True)

if __name__ == '__main__':
    run_pipeline()
