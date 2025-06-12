import pandas as pd
import joblib
import logging
import argparse

from src.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def make_predictions(input_path: str, model_path: str, fe_path: str, output_path: str) -> None:
    logging.info(f"Carregando modelo de: {model_path}")
    model = joblib.load(model_path)
    
    logging.info(f"Carregando pipeline de features de: {fe_path}")
    fe_pipeline = joblib.load(fe_path)

    logging.info(f"Carregando dados de teste de: {input_path}")
    test_data = pd.read_csv(input_path)
    passenger_ids = test_data['PassengerId']
    
    logging.info("Aplicando engenharia de features nos dados de teste...")
    X_test_processed = fe_pipeline.transform(test_data)
    
    logging.info("Gerando previsões...")
    predictions = model.predict(X_test_processed)
    
    submission_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions.astype(int)})
    submission_df.to_csv(output_path, index=False)
    logging.info(f"Arquivo de submissão salvo em: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gera previsões para o desafio Titanic.")
    parser.add_argument("--input", type=str, default=str(settings.TEST_FILE))
    parser.add_argument("--output", type=str, default=str(settings.BASE_DIR / "submission.csv"))
    parser.add_argument("--model", type=str, default=str(settings.BEST_MODEL_FILE))
    parser.add_argument("--fe-pipeline", type=str, default=str(settings.FE_PIPELINE_FILE))
    
    args = parser.parse_args()
    make_predictions(args.input, args.model, args.fe_pipeline, args.output)
