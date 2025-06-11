import pandas as pd
import logging
from typing import List, Tuple

from src.config import settings
from src.models import train_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def run_pipeline() -> None:
    logger.info("="*50)
    logger.info("INICIANDO PIPELINE DE TREINAMENTO DO MODELO TITANIC")
    logger.info("="*50)
    
    try:
        train_model.main()
        logger.info("Pipeline de treinamento concluído com sucesso.")
    except Exception as e:
        logger.error(f"Ocorreu um erro ao executar o pipeline de treinamento: {e}", exc_info=True)

if __name__ == '__main__':
    run_pipeline()
