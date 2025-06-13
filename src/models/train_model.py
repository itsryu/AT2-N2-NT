import pandas as pd
import joblib
import logging
import os
import numpy as np
import json
import time
import warnings
import sys
import optuna
import argparse

from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from src.config import settings
from src.processing.preprocessor import feature_engineering

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

TrainResult = Tuple[str, Pipeline, float, float, Dict[str, Any]]

def create_modeling_pipeline(model: Any) -> Pipeline:
    numeric_features = ["Age", "Fare", "FamilySize"]
    categorical_features = [
        "Embarked",
        "Sex",
        "Pclass",
        "Title",
        "IsAlone",
        "AgeGroup",
        "FareBin",
    ]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def get_objective_function(model_name: str, X: pd.DataFrame, y: pd.Series, models_to_tune: Dict[str, Any]):
    def objective(trial: optuna.Trial) -> float:
        if model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
            model_instance = models_to_tune[model_name](random_state=42, n_jobs=1, **params)
        elif model_name == "GradientBoosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
            }
            model_instance = models_to_tune[model_name](random_state=42, **params)
        elif model_name == "SVC":
            params = {"C": trial.suggest_float("C", 0.1, 10.0)}
            model_instance = models_to_tune[model_name](random_state=42, probability=True, **params)
        elif model_name == "XGBClassifier":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
            }
            model_instance = models_to_tune[model_name](random_state=42, eval_metric="logloss", **params)
        elif model_name == "LGBMClassifier":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 50),
            }
            model_instance = models_to_tune[model_name](random_state=42, verbose=-1, **params)
        else:
            raise ValueError(f"Modelo {model_name} não suportado.")

        pipeline = create_modeling_pipeline(model_instance)
        score = cross_val_score(
            pipeline, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="accuracy"
        ).mean()
        return score
    return objective

def train_single_model(model_name: str, X: pd.DataFrame, y: pd.Series, n_trials: int, models_to_tune: Dict[str, Any]) -> TrainResult:
    study = optuna.create_study(direction="maximize")
    objective_fn = get_objective_function(model_name, X, y, models_to_tune)
    study.optimize(objective_fn, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    best_params = study.best_params
    logger.info(f"Otimização para {model_name} concluída. Melhores parâmetros: {best_params}")

    model_class = models_to_tune[model_name]
    if model_name == "SVC":
        best_params["probability"] = True
    elif model_name == "LGBMClassifier":
        best_params["verbose"] = -1

    final_model_instance = model_class(**best_params, random_state=42)
    final_pipeline = create_modeling_pipeline(final_model_instance)

    cv_scores = cross_val_score(
        final_pipeline, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="accuracy"
    )
    mean_accuracy, std_dev = np.mean(cv_scores), np.std(cv_scores)
    
    logger.info(f"Modelo {model_name} validado. Acurácia: {mean_accuracy:.4f} (+/- {std_dev:.4f})")
    final_pipeline.fit(X, y)
    
    return model_name, final_pipeline, mean_accuracy, std_dev, best_params

def main(models_to_train: List[str], n_trials: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    start_time = time.perf_counter()
    training_metrics: Dict[str, Any] = {"individual_models": {}}

    df = pd.read_csv(str(settings.TRAIN_FILE))
    logger.info("Aplicando engenharia de features...")
    
    fe_pipeline = Pipeline(steps=[('feature_engineering', FunctionTransformer(feature_engineering))])
    
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_processed = fe_pipeline.fit_transform(X)
    
    joblib.dump(fe_pipeline, settings.FE_PIPELINE_FILE)
    logger.info(f"Pipeline de engenharia de features salvo em: {settings.FE_PIPELINE_FILE}")

    models_to_tune_filtered = {k: v for k, v in settings.MODELS_TO_TUNE.items() if k in models_to_train}
    logger.info(f"Modelos a serem treinados: {list(models_to_tune_filtered.keys())}")
    logger.info(f"Número de trials por modelo: {n_trials}")

    results: List[TrainResult] = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(train_single_model, name, X_processed.copy(), y.copy(), n_trials, models_to_tune_filtered): name
            for name in models_to_tune_filtered
        }
        for future in as_completed(futures):
            results.append(future.result())

    if not results:
        logger.error("Nenhum modelo foi treinado. Verifique as configurações.")
        return

    top_models: List[TrainResult] = sorted(results, key=lambda item: item[2], reverse=True)[:3]

    for name, pipeline, score, std_dev, params in top_models:
        model_path = settings.MODELS_DIR / f"{name.lower()}_model.joblib"
        joblib.dump(pipeline, model_path)
        training_metrics["individual_models"][name] = {"accuracy": score, "std_dev": std_dev, "params": params, "path": str(model_path)}
        logger.info(f"Modelo {name} salvo com acurácia {score:.4f} (+/- {std_dev:.4f}).")
    
    logger.info("Iniciando treinamento do modelo Stacking Ensemble...")
    
    stacking_estimators = [(name.lower(), model) for name, model, _, _, _ in top_models]
    meta_model = LogisticRegression(random_state=42)
    stacking_clf = StackingClassifier(estimators=stacking_estimators, final_estimator=meta_model, cv=5, n_jobs=-1)

    cv_scores = cross_val_score(stacking_clf, X_processed, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="accuracy", n_jobs=-1)
    mean_score, std_score = np.mean(cv_scores), np.std(cv_scores)
    logger.info(f"Acurácia do Stacking: {mean_score:.4f} (+/- {std_score:.4f})")

    training_metrics["ensemble_model"] = {"type": "StackingClassifier", "accuracy": mean_score, "std_dev": std_score, "estimators": [name for name, _, _, _, _ in top_models], "final_estimator": type(meta_model).__name__}

    logger.info("Treinando o Stacking final com todos os dados...")
    stacking_clf.fit(X_processed, y)
    joblib.dump(stacking_clf, settings.BEST_MODEL_FILE)
    logger.info(f"Melhor modelo (Ensemble) salvo em: {settings.BEST_MODEL_FILE}")

    training_metrics["total_training_time"] = time.perf_counter() - start_time
    with open(settings.METRICS_FILE, "w", encoding='utf-8') as f:
        json.dump(training_metrics, f, indent=4)

    logger.info(f"Métricas de treinamento salvas em: {settings.METRICS_FILE}")
    logger.info("Processo de treinamento concluído.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina modelos de classificação para o Titanic.")
    parser.add_argument("--models", type=str, default=",".join(settings.MODELS_TO_TUNE.keys()), help="Nomes dos modelos a treinar, separados por vírgula.")
    parser.add_argument("--n_trials", type=int, default=50, help="Número de tentativas de otimização do Optuna.")
    args = parser.parse_args()
    
    models_list = [model.strip() for model in args.models.split(',')]
    main(models_to_train=models_list, n_trials=args.n_trials)
