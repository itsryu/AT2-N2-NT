import streamlit as st
import pandas as pd
import joblib
import sys
import os
import json
import glob
import subprocess
from typing import Any, List

# Adiciona o diretório raiz ao path para importações corretas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import settings
from src.processing.preprocessor import advanced_feature_engineering

st.set_page_config(page_title="Dashboard Titanic", page_icon="🚢", layout="wide")

def check_artifacts_exist() -> bool:
    """Verifica se os artefatos essenciais do modelo existem."""
    return (
        os.path.exists(settings.BEST_MODEL_FILE) and
        os.path.exists(settings.METRICS_FILE) and
        os.path.exists(settings.FE_PIPELINE_FILE)
    )

@st.cache_data
def load_passenger_data() -> pd.DataFrame:
    """Carrega os dados dos passageiros do arquivo de treino."""
    return pd.read_csv(settings.TRAIN_FILE)

if not check_artifacts_exist():
    st.title("🚢 Modelos do Titanic Não Encontrados")
    st.info("Os modelos de machine learning ainda não foram treinados. Configure e inicie o treinamento abaixo.")

    with st.expander("⚙️ Configurações de Treinamento", expanded=True):
        available_models = list(settings.MODELS_TO_TUNE.keys())
        selected_models = st.multiselect(
            "Selecione os modelos para treinar:",
            options=available_models,
            default=available_models
        )
        n_trials = st.number_input("Número de tentativas de otimização (Optuna):", min_value=1, max_value=200, value=50)

    if st.button("Treinar Modelos", type="primary"):
        if not selected_models:
            st.error("Por favor, selecione pelo menos um modelo para treinar.")
        else:
            st.info("O treinamento foi iniciado. Por favor, aguarde...")
            
            with st.expander("Ver Logs de Treinamento", expanded=True):
                log_placeholder = st.empty()
                log_output = ""
                
                # Constrói o comando para o subprocesso com os argumentos
                command = [
                    sys.executable, "-u", "-m", "src.models.train_model",
                    "--models", ",".join(selected_models),
                    "--n_trials", str(n_trials)
                ]
                
                process = subprocess.Popen(
                    command,
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
        """Carrega um artefato a partir de um arquivo .joblib ou .json."""
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

    # Carregamento de dados e artefatos
    passenger_df = load_passenger_data()
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

            # Seleção de passageiro com preenchimento automático
            passenger_list = ["Digitar manualmente"] + passenger_df['Name'].tolist()
            selected_name = st.sidebar.selectbox("Selecione um passageiro (opcional):", passenger_list)

            # Define valores padrão ou do passageiro selecionado
            if selected_name != "Digitar manualmente":
                p_data = passenger_df[passenger_df['Name'] == selected_name].iloc[0]
                sex_default = 0 if p_data['Sex'] == 'male' else 1
                age_default = int(p_data['Age']) if pd.notna(p_data['Age']) else 29
                sibsp_default = int(p_data['SibSp'])
                parch_default = int(p_data['Parch'])
                fare_default = float(p_data['Fare'])
                pclass_default = int(p_data['Pclass'])
                embarked_default = ['S', 'C', 'Q'].index(p_data['Embarked']) if pd.notna(p_data['Embarked']) else 0
            else:
                p_data = None
                sex_default, age_default, sibsp_default, parch_default, fare_default, pclass_default, embarked_default = 0, 29, 0, 0, 32.2, 3, 0

            # Widgets com valores preenchidos
            sex = st.sidebar.selectbox("Sexo", ('male', 'female'), index=sex_default, key='sex')
            age = st.sidebar.slider("Idade", 0, 100, age_default, key='age')
            sibsp = st.sidebar.slider("Nº de Irmãos/Cônjuges", 0, 8, sibsp_default, key='sibsp')
            parch = st.sidebar.slider("Nº de Pais/Filhos", 0, 6, parch_default, key='parch')
            fare = st.sidebar.number_input("Tarifa", 0.0, 600.0, fare_default, key='fare')
            pclass = st.sidebar.selectbox("Classe", (1, 2, 3), index=pclass_default - 1, key='pclass')
            embarked = st.sidebar.selectbox("Porto de Embarque", ('S', 'C', 'Q'), index=embarked_default, key='embarked')
            
            # Usa o nome selecionado ou o manual
            name_to_use = selected_name if p_data is not None else "Manual Input"

            if st.button("Realizar Predição com " + model_choice, type="primary"):
                input_data = {
                    'PassengerId': [0], 'Pclass': [pclass], 'Name': [name_to_use], 'Sex': [sex],
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
