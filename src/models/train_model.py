import pandas as pd
import joblib
import logging
import os
import numpy as np
import json
import time
import warnings
import optuna

from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from src.config import settings
from src.processing.preprocessor import advanced_feature_engineering

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

TrainResult = Tuple[str, Pipeline, float, Dict[str, Any]]


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


def get_objective_function(model_name: str, X: pd.DataFrame, y: pd.Series):
    def objective(trial: optuna.Trial) -> float:
        if model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "criterion": trial.suggest_categorical(
                    "criterion", ["gini", "entropy"]
                ),
            }
            model_instance = settings.MODELS_TO_TUNE[model_name](
                random_state=42, n_jobs=1, **params
            )
        elif model_name == "GradientBoosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
            }
            model_instance = settings.MODELS_TO_TUNE[model_name](
                random_state=42, **params
            )
        elif model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "gamma": trial.suggest_float("gamma", 0, 0.5),
            }
            model_instance = settings.MODELS_TO_TUNE[model_name](
                random_state=42, eval_metric="logloss", n_jobs=1, **params
            )
        elif model_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 50),
            }
            model_instance = settings.MODELS_TO_TUNE[model_name](
                random_state=42, n_jobs=1, verbose=-1, **params
            )
        elif model_name == "SVC":
            params = {
                "C": trial.suggest_float("C", 0.1, 100, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            }
            model_instance = settings.MODELS_TO_TUNE[model_name](
                probability=True, random_state=42, kernel="rbf", **params
            )
        else:
            raise ValueError(f"Modelo desconhecido para otimização: {model_name}")

        pipeline = create_modeling_pipeline(model_instance)
        return cross_val_score(
            pipeline, X, y, cv=StratifiedKFold(n_splits=5), scoring="accuracy", n_jobs=1
        ).mean()

    return objective


def tune_and_train_model(
    model_name: str, X: pd.DataFrame, y: pd.Series, n_trials: int = 50
) -> TrainResult:
    logging.info(f"[Optuna] Iniciando otimização para: {model_name}")
    objective_func = get_objective_function(model_name, X, y)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=n_trials, n_jobs=1)

    best_score, best_params = study.best_value, study.best_params
    logging.info(
        f"[Optuna] Otimização para {model_name} concluída. Melhor acurácia: {best_score:.4f}"
    )

    model_class = settings.MODELS_TO_TUNE[model_name]

    if model_name == "SVC":
        best_params["probability"] = True

    final_model_instance = model_class(**best_params, random_state=42)
    final_pipeline = create_modeling_pipeline(final_model_instance)
    final_pipeline.fit(X, y)
    return model_name, final_pipeline, best_score, best_params


def main() -> None:
    start_time = time.perf_counter()
    training_metrics: Dict[str, Any] = {"individual_models": {}}

    df = pd.read_csv(str(settings.TRAIN_FILE))

    logging.info(
        "Aplicando engenharia de features robusta e calculando limites de binning..."
    )

    try:
        _, fare_bins = pd.qcut(
            df["Fare"].fillna(df["Fare"].median()), 4, retbins=True, duplicates="drop"
        )
    except ValueError:
        _, fare_bins = pd.cut(df["Fare"].fillna(df["Fare"].median()), 4, retbins=True)

    bins_to_save = {"AgeBins": [0, 12, 18, 60, np.inf], "FareBins": fare_bins.tolist()}
    training_metrics["binning_thresholds"] = bins_to_save

    df_processed = advanced_feature_engineering(df, bins=bins_to_save)

    df_processed.to_csv(settings.PROCESSED_TRAIN_FILE, index=False)
    logging.info(f"Dados processados salvos em: {settings.PROCESSED_TRAIN_FILE}")

    X = df_processed.drop(columns=[settings.TARGET_COLUMN])
    y = df_processed[settings.TARGET_COLUMN]

    all_models: List[TrainResult] = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        logging.info(
            f"Iniciando otimização paralela com Optuna em {executor._max_workers} threads..."
        )
        future_to_model = {
            executor.submit(tune_and_train_model, name, X, y): name
            for name in settings.MODELS_TO_TUNE.keys()
        }

        for future in as_completed(future_to_model):
            try:
                name, model, score, params = future.result()
                all_models.append((name, model, score, params))
                joblib.dump(model, settings.ARTIFACTS_DIR / f"{name}_model.joblib")
                training_metrics["individual_models"][name] = {
                    "accuracy": score,
                    "best_params": params,
                }
            except Exception as e:
                logging.error(
                    f"Erro ao otimizar o modelo {future_to_model[future]}: {e}",
                    exc_info=True,
                )

    if not all_models:
        logging.error("Nenhum modelo foi otimizado com sucesso.")
        return

    all_models.sort(key=lambda x: x[2], reverse=True)
    top_models = all_models[:3]

    logging.info("-" * 50)
    logging.info("Melhores Modelos Individuais (Base para Stacking):")
    for name, _, score, _ in top_models:
        logging.info(f"  - {name}: {score:.4f}")

    logging.info("Criando e avaliando o ensemble com StackingClassifier...")
    stacking_estimators = [(name, model) for name, model, _, _ in top_models]
    meta_model = LogisticRegression(solver="liblinear", random_state=42)
    stacking_clf = StackingClassifier(
        estimators=stacking_estimators, final_estimator=meta_model, cv=5, n_jobs=-1
    )

    cv_scores = cross_val_score(
        stacking_clf,
        X,
        y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
    )
    mean_score, std_score = np.mean(cv_scores), np.std(cv_scores)
    logging.info(
        f"Acurácia do Stacking (Validação Cruzada): {mean_score:.4f} (+/- {std_score:.4f})"
    )

    training_metrics["ensemble_model"] = {
        "type": "StackingClassifier",
        "accuracy": mean_score,
        "std_dev": std_score,
        "estimators": [name for name, _, _, _ in top_models],
        "final_estimator": type(meta_model).__name__,
    }

    logging.info("Treinando o Stacking final com todos os dados...")
    stacking_clf.fit(X, y)

    joblib.dump(stacking_clf, settings.BEST_MODEL_FILE)
    logging.info(
        f"Melhor modelo (Stacking Ensemble) salvo em: {settings.BEST_MODEL_FILE}"
    )

    end_time = time.perf_counter()
    duration = end_time - start_time
    training_metrics["total_training_time_seconds"] = duration
    logging.info(f"Tempo total de treinamento: {duration:.2f} segundos.")

    with open(settings.METRICS_FILE, "w") as f:
        json.dump(training_metrics, f, indent=4)
    logging.info(f"Métricas de treinamento salvas em: {settings.METRICS_FILE}")


main()
