import streamlit as st
import pandas as pd
import joblib
import sys
import os
import json
from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import settings
from src.models.train_model import advanced_feature_engineering

st.set_page_config(page_title="Previsão de Sobrevivência - Titanic", page_icon="🚢", layout="wide")

@st.cache_resource
def load_artifact(path: str) -> Any:
    if not os.path.exists(path):
        st.error(f"Artefato não encontrado em '{path}'. Execute o pipeline de treinamento primeiro.")
        return None
        
    try:
        if path.endswith(".json"):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            return joblib.load(path)
    except Exception as e:
        st.error(f"Erro ao carregar o artefato de '{path}': {e}")
        return None

model = load_artifact(str(settings.BEST_MODEL_FILE))
metrics = load_artifact(str(settings.METRICS_FILE))
binning_thresholds = metrics.get("binning_thresholds") if metrics else None

st.title("🚢 Previsão de Sobrevivência no Titanic")
st.markdown("### Pipeline Campeão com Engenharia de Features Robusta e Stacking")

st.sidebar.header("Informações do Passageiro")

def user_input_features() -> pd.DataFrame:
    pclass = st.sidebar.selectbox("Classe da Passagem (Pclass)", (1, 2, 3))
    name = st.sidebar.text_input("Nome Completo", "Mr. John Doe")
    sex = st.sidebar.selectbox("Sexo", ("male", "female"))
    age = st.sidebar.slider("Idade (Age)", 0, 100, 29, help="O modelo irá estimar este valor se ausente, mas um valor real melhora a predição.")
    sibsp = st.sidebar.slider("Nº de Irmãos/Cônjuges (SibSp)", 0, 8, 0)
    parch = st.sidebar.slider("Nº de Pais/Filhos (Parch)", 0, 6, 0)
    fare = st.sidebar.number_input("Tarifa (Fare)", 0.0, 600.0, 32.2)
    embarked = st.sidebar.selectbox("Porto de Embarque (Embarked)", ('S', 'C', 'Q'))
    
    data = {
        'PassengerId': [0], 'Pclass': [pclass], 'Name': [name], 'Sex': [sex],
        'Age': [age], 'SibSp': [sibsp], 'Parch': [parch], 'Ticket': [''],
        'Fare': [fare], 'Cabin': [None], 'Embarked': [embarked]
    }
    return pd.DataFrame(data)

if model and binning_thresholds:
    input_df_raw = user_input_features()
    st.subheader("Dados do Passageiro Inseridos (Brutos):")
    st.write(input_df_raw[['Name', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']])

    if st.button("Realizar Previsão", type="primary"):
        input_df_processed = advanced_feature_engineering(input_df_raw, bins=binning_thresholds)

        st.subheader("Dados Processados (Enviados ao Modelo):")
        st.write(input_df_processed[['Title', 'FamilySize', 'IsAlone', 'AgeGroup', 'FareBin']])
        
        try:
            prediction = model.predict(input_df_processed)
            prediction_proba = model.predict_proba(input_df_processed)
            survival_probability = prediction_proba[0][1]

            st.subheader("Resultado da Previsão")
            if prediction[0] == 1:
                st.success(f"**Provavelmente Sobreviveria** (Probabilidade: {survival_probability:.2%})")
            else:
                st.error(f"**Provavelmente NÃO Sobreviveria** (Probabilidade de Sobrevivência: {survival_probability:.2%})")
            st.progress(survival_probability)
        except Exception as e:
            st.error(f"Ocorreu um erro durante a predição: {e}")

else:
    st.warning("Um ou mais artefatos do modelo não foram encontrados. Execute `python -m src.main` para treinar o pipeline campeão.")
