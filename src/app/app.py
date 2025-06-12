import streamlit as st
import pandas as pd
import joblib
import sys
import os
import json
import glob
from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import settings
from src.models.train_model import advanced_feature_engineering

st.set_page_config(page_title="Dashboard Titanic", page_icon="🚢", layout="wide")

@st.cache_resource
def load_artifact(path: str) -> Any:
    """Carrega um artefato a partir de um arquivo .joblib ou .json."""
    if not os.path.exists(path):
        return None
    try:
        if path.endswith(".json"):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            return joblib.load(path)
    except Exception:
        return None

metrics = load_artifact(str(settings.METRICS_FILE))

st.sidebar.title("🚢 Painel de Navegação")
page = st.sidebar.radio("Selecione uma página:", ["Análise Exploratória (EDA)", "Métricas e Predição"])

if page == "Análise Exploratória (EDA)":
    st.title("📊 Análise Exploratória dos Dados do Titanic")
    st.markdown("Esta página exibe as visualizações geradas durante a fase de EDA para entender os padrões nos dados.")
    
    figure_paths = sorted(glob.glob(os.path.join(settings.FIGURES_DIR, "*.png")))
    
    if not figure_paths:
        st.warning(f"Nenhuma figura encontrada em `{settings.FIGURES_DIR}`. Execute o script da EDA primeiro: `python -m scripts.titanic_eda_professional`")
    else:
        for fig_path in figure_paths:
            title = os.path.basename(fig_path).replace('_', ' ').replace('.png', '').title()[3:]
            with st.expander(f"▶️ {title}", expanded=False):
                st.image(fig_path)

elif page == "Métricas e Predição":
    st.title("⚙️ Métricas dos Modelos e Predição Interativa")
    
    if not metrics:
        st.error(f"Ficheiro de métricas não encontrado em `{settings.METRICS_FILE}`. Execute o pipeline de treinamento principal: `python -m src.main`")
    else:
        st.sidebar.divider()
        st.sidebar.header("Seleção de Modelo")
        
        available_models = ["Ensemble Campeão"] + list(metrics["individual_models"].keys())
        model_choice = st.sidebar.selectbox("Escolha um modelo para inspecionar e usar para predição:", available_models)

        st.header(f"Métricas para: `{model_choice}`")
        
        if model_choice == "Ensemble Campeão":
            model_data = metrics["ensemble_model"]
            model_path = str(settings.BEST_MODEL_FILE)
            st.metric(label="Acurácia (Validação Cruzada)", value=f"{model_data['accuracy']:.4f}")
            st.metric(label="Estabilidade (Desvio Padrão)", value=f"± {model_data['std_dev']:.4f}")
            st.info(f"**Tipo:** {model_data['type']} | **Estimadores Base:** {', '.join(model_data['estimators'])}")
        else:
            model_data = metrics["individual_models"][model_choice]
            model_path = str(settings.ARTIFACTS_DIR / f"{model_choice}_model.joblib")
            st.metric(label="Acurácia (Validação Cruzada)", value=f"{model_data['accuracy']:.4f}")
            with st.expander("Ver Melhores Hiperparâmetros (Optuna)"):
                st.json(model_data['best_params'])
        
        st.divider()
        st.header("Faça uma Nova Predição")
        
        model = load_artifact(model_path)
        if not model:
            st.error(f"Não foi possível carregar o ficheiro do modelo: {model_path}")
        else:
            pclass = st.sidebar.selectbox("Classe da Passagem (Pclass)", (1, 2, 3), key='pclass')
            name = st.sidebar.text_input("Nome Completo", "Mr. John Doe", key='name')
            sex = st.sidebar.selectbox("Sexo", ("male", "female"), key='sex')
            age = st.sidebar.slider("Idade (Age)", 0, 100, 29, key='age')
            sibsp = st.sidebar.slider("Nº de Irmãos/Cônjuges (SibSp)", 0, 8, 0, key='sibsp')
            parch = st.sidebar.slider("Nº de Pais/Filhos (Parch)", 0, 6, 0, key='parch')
            fare = st.sidebar.number_input("Tarifa (Fare)", 0.0, 600.0, 32.2, key='fare')
            embarked = st.sidebar.selectbox("Porto de Embarque (Embarked)", ('S', 'C', 'Q'), key='embarked')

            if st.button("Realizar Predição com " + model_choice, type="primary"):
                input_data = {
                    'PassengerId': [0], 'Pclass': [pclass], 'Name': [name], 'Sex': [sex],
                    'Age': [age], 'SibSp': [sibsp], 'Parch': [parch], 'Ticket': [''],
                    'Fare': [fare], 'Cabin': [None], 'Embarked': [embarked]
                }

                input_df = pd.DataFrame(input_data)
                processed_df = advanced_feature_engineering(input_df)
                
                prediction = model.predict(processed_df)
                prediction_proba = model.predict_proba(processed_df)
                survival_probability = prediction_proba[0][1]

                if prediction[0] == 1:
                    st.success(f"**Resultado:** Passageiro Provavelmente **SOBREVIVERIA** (Probabilidade: {survival_probability:.2%})")
                else:
                    st.error(f"**Resultado:** Passageiro Provavelmente **NÃO SOBREVIVERIA** (Probabilidade de Sobrevivência: {survival_probability:.2%})")
