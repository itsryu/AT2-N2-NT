import pandas as pd
import numpy as np
import joblib
import logging
import os
from typing import Dict, Any, List, Tuple, Type
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() 
    ]
)

Model = Any
ModelParams = Dict[str, Any]
TrainResult = Tuple[Model, float, str]

DATA_PATH = '../../data/raw/train.csv'
ARTIFACTS_PATH = '../../artifacts'
os.makedirs(ARTIFACTS_PATH, exist_ok=True) 

MODELS_TO_TRAIN: Dict[str, Tuple[Model, ModelParams]] = {
    "LogisticRegression": (
        LogisticRegression(solver='liblinear', random_state=42),
        {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__penalty": ["l1", "l2"]
        }
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_leaf": [1, 2, 4]
        }
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 5]
        }
    ),
    "SVC": (
        SVC(probability=True, random_state=42),
        {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__gamma": ['scale', 'auto'],
            "classifier__kernel": ['rbf', 'linear']
        }
    )
}

def load_data(path: str) -> pd.DataFrame:
    logging.info(f"Carregando dados de: {path}")
    return pd.read_csv(path)

def create_pipeline(model: Model) -> Pipeline:
    
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler()), 
        ('classifier', model) 
    ])

def train_single_model(model_name: str, model: Model, params: ModelParams, X: pd.DataFrame, y: pd.Series) -> TrainResult:
    logging.info(f"[Thread] Iniciando treinamento para: {model_name}")
    pipeline = create_pipeline(model)
    
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=cv,
        scoring='accuracy',
        n_jobs=1 
    )
    
    grid_search.fit(X, y)
    
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    
    logging.info(f"[Thread] Treinamento para {model_name} concluído. Melhor acurácia: {best_score:.4f}")
    
    return best_model, best_score, model_name

def main() -> None:
    df = load_data(DATA_PATH)
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    target = 'Survived'
    
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].fillna('S').map({'S': 0, 'C': 1, 'Q': 2})
    features.extend(['Sex', 'Embarked'])

    X = df[features]
    y = df[target]
    
    results: List[TrainResult] = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        logging.info(f"Iniciando treinamento paralelo em {executor._max_workers} threads...")
        
        future_to_model = {
            executor.submit(train_single_model, name, model, params, X, y): name
            for name, (model, params) in MODELS_TO_TRAIN.items()
        }
        
        for future in as_completed(future_to_model):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                model_name = future_to_model[future]
                logging.error(f"Erro ao treinar o modelo {model_name}: {e}")

    if not results:
        logging.error("Nenhum modelo foi treinado com sucesso.")
        return

    best_model, best_score, best_model_name = max(results, key=lambda item: item[1])
    
    logging.info("-" * 50)
    logging.info(f"Treinamento concluído!")
    logging.info(f"Melhor modelo: {best_model_name}")
    logging.info(f"Acurácia (validação cruzada): {best_score:.4f}")
    
    
    model_path = os.path.join(ARTIFACTS_PATH, 'best_model.joblib')
    joblib.dump(best_model, model_path)
    logging.info(f"Melhor modelo salvo em: {model_path}")

if __name__ == '__main__':
    main()
