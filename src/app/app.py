import streamlit as st
import pandas as pd
import joblib
import sys
import os
import json
import glob
import subprocess
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import settings
from src.processing.preprocessor import advanced_feature_engineering

st.set_page_config(page_title="Dashboard Titanic", page_icon="🚢", layout="wide")

def check_artifacts_exist() -> bool:
    return (
        os.path.exists(settings.BEST_MODEL_FILE) and
        os.path.exists(settings.METRICS_FILE) and
        os.path.exists(settings.FE_PIPELINE_FILE)
    )

if not check_artifacts_exist():
    st.title("🚢 Modelos do Titanic Não Encontrados")
    st.info("Os modelos de machine learning ainda não foram treinados. Clique no botão abaixo para iniciar o treinamento.")

    if st.button("Treinar Modelos", type="primary"):
        st.info("O treinamento foi iniciado. Por favor, aguarde...")
        
        with st.expander("Ver Logs de Treinamento", expanded=True):
            log_placeholder = st.empty()
            log_output = ""
            
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "src.models.train_model"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='cp1252',
                errors='ignore',
                bufsize=1
            )

            for line in iter(process.stdout.readline, ''):
                log_output += line
                log_placeholder.code(log_output, language='log')

            process.wait()

        if process.returncode == 0:
            st.success("Treinamento concluído com sucesso! A página será recarregada.")
            st.balloons()
            if st.button("Recarregar Página"):
                 st.rerun()
        else:
            st.error("Ocorreu um erro durante o treinamento. Verifique os logs acima.")
else:
    @st.cache_resource
    def load_artifact(path: str) -> Any:
        if not os.path.exists(path):
            st.error(f"Artefato não encontrado em: {path}")
            return None
        try:
            if path.endswith(".json"):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return joblib.load(path)
        except Exception as e:
            st.error(f"Erro ao carregar o artefato {path}: {e}")
            return None

    metrics = load_artifact(str(settings.METRICS_FILE))
    
    st.sidebar.title("🚢 Painel de Navegação")
    page = st.sidebar.radio("Selecione uma página:", ["Métricas e Predição", "Análise Exploratória (EDA)"])

    if page == "Análise Exploratória (EDA)":
        st.title("📊 Análise Exploratória dos Dados do Titanic")
        st.markdown("Esta página exibe as visualizações geradas durante a fase de EDA para entender os padrões nos dados.")
        
        figure_paths = sorted(glob.glob(os.path.join(settings.FIGURES_DIR, "*.html")))
        if not figure_paths:
            st.warning("Nenhuma figura da Análise Exploratória foi encontrada.")
        else:
            for fig_path in figure_paths:
                fig_name = os.path.basename(fig_path).replace(".html", "").replace("_", " ").title()
                with st.expander(fig_name, expanded=False):
                    with open(fig_path, 'r', encoding='utf-8') as f:
                        st.components.v1.html(f.read(), height=500, scrolling=True)

    elif page == "Métricas e Predição":
        st.title("📈 Métricas de Modelos e Predição de Sobrevivência")

        if metrics:
            st.header("🎯 Performance dos Modelos")
            
            for model_name, data in metrics.get("individual_models", {}).items():
                with st.container():
                    col1, _, _ = st.columns(3)
                    col1.metric(label=f"Modelo: **{model_name}**", value=f"{data['accuracy']:.4f}", delta=f"± {data.get('std_dev', 0.0):.4f}", help="Acurácia e desvio padrão da validação cruzada.")
                    
            if "ensemble_model" in metrics:
                ens = metrics["ensemble_model"]
                st.subheader("🚀 Modelo Ensemble (Stacking)")
                col1, col2 = st.columns(2)
                col1.metric(label="Acurácia (Ensemble)", value=f"{ens['accuracy']:.4f}", delta=f"± {ens.get('std_dev', 0.0):.4f}")
                col2.info(f"**Estimadores Base:** {', '.join(ens['estimators'])}")

        st.header("🔮 Realizar Nova Predição")

        models = load_artifact(str(settings.BEST_MODEL_FILE))
        fe_pipeline = load_artifact(str(settings.FE_PIPELINE_FILE))
        
        if models and fe_pipeline:
            model_options = list(metrics.get("individual_models", {}).keys())
            if "ensemble_model" in metrics:
                model_options.append("Ensemble (Stacking)")
            
            model_choice = st.selectbox("Escolha um modelo para predição:", model_options)

            if model_choice == "Ensemble (Stacking)":
                model = models
            else:
                model = models.named_estimators_[model_choice.lower()]

            st.sidebar.header("Parâmetros do Passageiro")
            name = st.sidebar.text_input("Nome", "John Doe", key='name')
            sex = st.sidebar.selectbox("Sexo", ('male', 'female'), key='sex')
            age = st.sidebar.slider("Idade", 0, 100, 29, key='age')
            sibsp = st.sidebar.slider("Nº de Irmãos/Cônjuges", 0, 8, 0, key='sibsp')
            parch = st.sidebar.slider("Nº de Pais/Filhos", 0, 6, 0, key='parch')
            fare = st.sidebar.number_input("Tarifa", 0.0, 600.0, 32.2, key='fare')
            pclass = st.sidebar.selectbox("Classe", (1, 2, 3), key='pclass')
            embarked = st.sidebar.selectbox("Porto de Embarque", ('S', 'C', 'Q'), key='embarked')

            if st.button("Realizar Predição com " + model_choice, type="primary"):
                input_data = {
                    'PassengerId': [0], 'Pclass': [pclass], 'Name': [name], 'Sex': [sex],
                    'Age': [age], 'SibSp': [sibsp], 'Parch': [parch], 'Ticket': [''],
                    'Fare': [fare], 'Cabin': [None], 'Embarked': [embarked]
                }

                input_df = pd.DataFrame(input_data)
                
                processed_df = fe_pipeline.transform(input_df)
                
                prediction = model.predict(processed_df)
                prediction_proba = model.predict_proba(processed_df)
                survival_probability = prediction_proba[0][1]

                if prediction[0] == 1:
                    st.success(f"**Resultado:** Passageiro Provavelmente **SOBREVIVERIA** (Probabilidade: {survival_probability:.2%})")
                else:
                    st.error(f"**Resultado:** Passageiro Provavelmente **NÃO SOBREVIVERIA** (Probabilidade de Sobrevivência: {survival_probability:.2%})")
        else:
            st.error("Não foi possível carregar os modelos ou o pipeline de features. Por favor, treine-os primeiro.")
