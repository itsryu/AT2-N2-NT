from pathlib import Path
from typing import Dict, Type

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC

Model = type

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" 

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train_processed.csv"
BEST_MODEL_FILE = ARTIFACTS_DIR / "ensemble_model.joblib"
FE_PIPELINE_FILE = ARTIFACTS_DIR / "fe_pipeline.joblib"
METRICS_FILE = ARTIFACTS_DIR / "training_metrics.json"

TARGET_COLUMN: str = "Survived"

MODELS_TO_TUNE: Dict[str, Model] = {
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
    "XGBClassifier": XGBClassifier,
    "LGBMClassifier": LGBMClassifier,
    "SVC": SVC,
}
