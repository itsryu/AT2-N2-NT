import pandas as pd
import numpy as np

def advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    # Extrair Título do Nome
    df_copy["Title"] = df_copy["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    common_titles = {"Mr", "Miss", "Mrs", "Master"}
    df_copy["Title"] = df_copy["Title"].apply(
        lambda x: x if x in common_titles else "Other"
    )

    # Criar Feature 'FamilySize' e 'IsAlone'
    df_copy["FamilySize"] = df_copy["SibSp"] + df_copy["Parch"] + 1
    df_copy["IsAlone"] = (df_copy["FamilySize"] == 1).astype(int)

    # Agrupar Idade em Bins
    df_copy["Age"] = df_copy["Age"].fillna(df_copy["Age"].median())
    df_copy["AgeGroup"] = pd.cut(
        df_copy["Age"],
        bins=[0, 12, 18, 60, np.inf],
        labels=["Child", "Teen", "Adult", "Senior"],
    )

    # Agrupar Tarifa em Bins
    df_copy["Fare"] = df_copy["Fare"].fillna(df_copy["Fare"].median())
    df_copy["FareBin"] = pd.qcut(
        df_copy["Fare"], 4, labels=["Low", "Mid", "High", "VeryHigh"], duplicates="drop"
    )

    # Preencher valores ausentes em 'Embarked'
    df_copy["Embarked"] = df_copy["Embarked"].fillna(df_copy["Embarked"].mode()[0])

    # Converter 'Sex' para numérico
    df_copy["Sex"] = df_copy["Sex"].map({"male": 0, "female": 1}).astype(int)

    return df_copy
