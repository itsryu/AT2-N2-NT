import logging

from src.models import train_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train() -> None:
    try:
        train_model.main(
            models_to_train=["RandomForest", "GradientBoosting", "XGBClassifier", "LGBMClassifier", "SVC"],
            n_trials=50
        )
    except Exception as e:
        logger.error(f"Ocorreu um erro fatal ao executar o pipeline de treinamento: {e}", exc_info=True)

if __name__ == '__main__':
    train()
