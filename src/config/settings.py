from pathlib import Path
from typing import List, Dict, Any, Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

Model = Any
ModelParams = Dict[str, Any]

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
SUBMISSION_FILE_EXAMPLE = RAW_DATA_DIR / "gender_submission.csv"
BEST_MODEL_FILE = ARTIFACTS_DIR / "best_model.joblib"

TARGET_COLUMN: str = "Survived" #

FEATURES: List[str] = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "FamilySize",
    "Has_Cabin",
    "Embarked_C",
    "Embarked_Q",
    "Embarked_S",
]

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
        }
    )
}
