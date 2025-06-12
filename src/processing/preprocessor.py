import pandas as pd
import numpy as np
from typing import Dict, Optional

def advanced_feature_engineering(df: pd.DataFrame, bins: Optional[Dict[str, list]] = None) -> pd.DataFrame:
    df_copy = df.copy()
    
    if 'Age' in df_copy.columns:
        df_copy['Age'].fillna(df_copy['Age'].median(), inplace=True)
    if 'Fare' in df_copy.columns:
        df_copy['Fare'].fillna(df_copy['Fare'].median(), inplace=True)
    if 'Embarked' in df_copy.columns:
        df_copy['Embarked'].fillna(df_copy['Embarked'].mode()[0], inplace=True)
    
    df_copy['Title'] = df_copy['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    common_titles = {"Mr", "Miss", "Mrs", "Master"}
    df_copy['Title'] = df_copy['Title'].apply(lambda x: x if x in common_titles else 'Other')

    df_copy['FamilySize'] = df_copy['SibSp'] + df_copy['Parch'] + 1
    df_copy['IsAlone'] = (df_copy['FamilySize'] == 1).astype(int)
    
    if bins:
        df_copy['AgeGroup'] = pd.cut(df_copy['Age'], bins=bins['AgeBins'], labels=[1, 2, 3, 4], include_lowest=True)
        df_copy['FareBin'] = pd.cut(df_copy['Fare'], bins=bins['FareBins'], labels=[1, 2, 3, 4], include_lowest=True)
    else:
        df_copy['AgeGroup'] = pd.cut(df_copy['Age'], bins=[0, 12, 18, 60, np.inf], labels=[1, 2, 3, 4], include_lowest=True)
        try:
            df_copy['FareBin'] = pd.qcut(df_copy['Fare'], 4, labels=[1, 2, 3, 4], duplicates='drop')
        except ValueError:
            df_copy['FareBin'] = pd.cut(df_copy['Fare'], 4, labels=[1, 2, 3, 4], include_lowest=True)

    df_copy['Sex'] = df_copy['Sex'].map({'male': 0, 'female': 1}).astype(int)
    return df_copy
