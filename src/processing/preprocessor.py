import pandas as pd
from typing import Tuple

class TitanicDataPreprocessor:
    def __init__(self, is_training: bool = True):
        self.is_training = is_training
        self.imputation_values: dict = {}

    def _impute_age(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.is_training:
            
            self.imputation_values['Age'] = df['Age'].median()
        
        
        df['Age'].fillna(self.imputation_values['Age'], inplace=True)
        return df

    def _impute_embarked(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.is_training:
            self.imputation_values['Embarked'] = df['Embarked'].mode()[0]
        
        df['Embarked'].fillna(self.imputation_values['Embarked'], inplace=True)
        return df
        
    def _impute_fare(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.is_training:
            self.imputation_values['Fare'] = df['Fare'].median()

        df['Fare'].fillna(self.imputation_values['Fare'], inplace=True)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)     
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['Has_Cabin'] = df['Cabin'].notnull().astype(int)
        
        df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked', drop_first=False)
        
        return df

    def _ensure_all_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {'Embarked_C', 'Embarked_Q', 'Embarked_S'}
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy = self._impute_age(df_copy)
        df_copy = self._impute_embarked(df_copy)
        df_copy = self._impute_fare(df_copy)
        df_copy = self._engineer_features(df_copy)
        df_copy = self._ensure_all_columns(df_copy)
        
        return df_copy
