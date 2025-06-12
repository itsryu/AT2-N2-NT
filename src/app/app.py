import streamlit as st
import pandas as pd
import joblib
import sys
import os
import json
import glob
import subprocess
import time
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import settings

st.set_page_config(page_title="Dashboard Titanic", page_icon="🚢", layout="wide")

def check_artifacts_exist() -> bool:
    return (
        os.path.exists(settings.BEST_MODEL_FILE) and
        os.path.exists(settings.METRICS_FILE) and
        os.path.exists(settings.FE_PIPELINE_FILE)
    )

@st.cache_data
def load_passenger_data() -> pd.DataFrame:
    return pd.read_csv(settings.TRAIN_FILE)

@st.cache_data
def delete_artifacts():
    files_to_delete = [
        settings.BEST_MODEL_FILE,
        settings.METRICS_FILE,
        settings.FE_PIPELINE_FILE
    ]
    st.toast("Iniciando a remoção dos artefatos...")
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError as e:
            st.error(f"Erro ao remover {os.path.basename(file_path)}: {e}")

    if os.path.exists(settings.MODELS_DIR):
        model_files = glob.glob(os.path.join(settings.MODELS_DIR, "*.joblib"))
        for file_path in model_files:
            try:
                os.remove(file_path)
            except OSError as e:
                st.error(f"Erro ao remover modelo {os.path.basename(file_path)}: {e}")
    st.toast("Artefatos removidos com sucesso!")


if not check_artifacts_exist():
    st.title("🚢 Treinamento dos Modelos")
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
                
                command = [
                    sys.executable, "-u", "-m", "src.models.train_model",
                    "--models", ",".join(selected_models),
                    "--n_trials", str(n_trials)
                ]
                
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='cp1252', errors='ignore', bufsize=1
                )

                for line in iter(process.stdout.readline, ''):
                    log_output += line
                    log_placeholder.code(log_output, language='log')

                process.wait()

            if process.returncode == 0:
                st.success("Treinamento concluído com sucesso! A página será recarregada.")
                st.balloons()
                time.sleep(1)
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

    passenger_df = load_passenger_data()
    metrics = load_artifact(str(settings.METRICS_FILE))
    model = load_artifact(str(settings.BEST_MODEL_FILE))
    fe_pipeline = load_artifact(str(settings.FE_PIPELINE_FILE))
    
    st.sidebar.title("🚢 Painel de Navegação")

    page = st.sidebar.radio("Selecione uma página:", [
        "Métricas e Predição",
        "Análise Exploratória (EDA)",
        "Gerar Submissão Kaggle"
    ])

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
        st.title("📈 Métricas e Predição de Sobrevivência")

        if metrics and model and fe_pipeline:
            st.header("🔮 Selecione o Modelo")
            
            model_options = list(metrics.get("individual_models", {}).keys())
            if "ensemble_model" in metrics:
                model_options.append("Ensemble (Stacking)")
            
            model_choice = st.selectbox("Escolha um modelo para ver suas métricas e usar para predição:", model_options)

            st.header("🎯 Performance do Modelo Selecionado")

            with st.container(border=True):
                if model_choice == "Ensemble (Stacking)":
                    ens_data = metrics["ensemble_model"]
                    col1, col2 = st.columns(2)
                    col1.metric(
                        label="Acurácia (Ensemble)", 
                        value=f"{ens_data['accuracy']:.4f}", 
                        delta=f"± {ens_data.get('std_dev', 0.0):.4f}",
                        help="Acurácia e desvio padrão da validação cruzada."
                    )
                    col2.info(f"**Estimadores Base:** {', '.join(ens_data['estimators'])}")
                else:
                    model_data = metrics.get("individual_models", {}).get(model_choice)
                    if model_data:
                        st.metric(
                            label=f"Acurácia: {model_choice}",
                            value=f"{model_data['accuracy']:.4f}",
                            delta=f"± {model_data.get('std_dev', 0.0):.4f}",
                            help="Acurácia e desvio padrão da validação cruzada."
                        )
                    else:
                        st.warning("Métricas não encontradas para o modelo selecionado.")

            if model_choice == "Ensemble (Stacking)":
                model_to_predict = model
            else:
                model_to_predict = model.named_estimators_[model_choice.lower()]

            st.sidebar.header("Parâmetros do Passageiro")
            
            passenger_list = ["Digitar manualmente"] + passenger_df['Name'].tolist()
            selected_name = st.sidebar.selectbox("Selecione um passageiro (opcional):", passenger_list)

            sex_map = {"Masculino": "male", "Feminino": "female"}
            pclass_map = {"1ª Classe": 1, "2ª Classe": 2, "3ª Classe": 3}
            embarked_map = {"Southampton": "S", "Cherbourg": "C", "Queenstown": "Q"}
            
            if selected_name != "Digitar manualmente":
                p_data = passenger_df[passenger_df['Name'] == selected_name].iloc[0]
                sex_val, age_default = p_data['Sex'], int(p_data['Age']) if pd.notna(p_data['Age']) else 29
                sibsp_default, parch_default = int(p_data['SibSp']), int(p_data['Parch'])
                fare_default, pclass_val = float(p_data['Fare']), int(p_data['Pclass'])
                embarked_val = p_data['Embarked'] if pd.notna(p_data['Embarked']) else 'S'
            else:
                p_data = None
                sex_val, age_default, sibsp_default, parch_default, fare_default = 'male', 29, 0, 0, 32.2
                pclass_val, embarked_val = 3, 'S'

            sex_options = list(sex_map.keys())
            pclass_options = list(pclass_map.keys())
            embarked_options = list(embarked_map.keys())
            
            sex_default_label = [k for k, v in sex_map.items() if v == sex_val][0]
            pclass_default_label = [k for k, v in pclass_map.items() if v == pclass_val][0]
            embarked_default_label = [k for k, v in embarked_map.items() if v == embarked_val][0]

            sex_default_index, pclass_default_index, embarked_default_index = sex_options.index(sex_default_label), pclass_options.index(pclass_default_label), embarked_options.index(embarked_default_label)

            sex_label = st.sidebar.selectbox("Sexo", sex_options, index=sex_default_index, key='sex')
            age = st.sidebar.slider("Idade", 0, 100, age_default, key='age')
            sibsp = st.sidebar.slider("Nº de Irmãos/Cônjuges", 0, 8, sibsp_default, key='sibsp', help="Número de irmãos ou cônjuges do passageiro a bordo.")
            parch = st.sidebar.slider("Nº de Pais/Filhos", 0, 6, parch_default, key='parch', help="Número de pais ou filhos do passageiro a bordo.")
            fare = st.sidebar.number_input("Valor da Tarifa (£)", 0.0, 600.0, fare_default, key='fare', help="Valor pago pelo bilhete em Libras Esterlinas (£) de 1912.")
            pclass_label = st.sidebar.selectbox("Classe do Bilhete", pclass_options, index=pclass_default_index, key='pclass')
            embarked_label = st.sidebar.selectbox("Porto de Embarque", embarked_options, index=embarked_default_index, key='embarked')
            
            if st.button(f"Realizar Predição com {model_choice}", type="primary", use_container_width=True):
                input_data = {
                    'PassengerId': [0], 'Pclass': [pclass_map[pclass_label]], 
                    'Name': [selected_name if p_data is not None else "Manual Input"], 
                    'Sex': [sex_map[sex_label]], 'Age': [age], 'SibSp': [sibsp], 
                    'Parch': [parch], 'Ticket': [''], 'Fare': [fare], 'Cabin': [None], 
                    'Embarked': [embarked_map[embarked_label]]
                }
                input_df = pd.DataFrame(input_data)
                
                processed_df = fe_pipeline.transform(input_df)
                
                prediction = model_to_predict.predict(processed_df)
                prediction_proba = model_to_predict.predict_proba(processed_df)
                survival_probability = prediction_proba[0][1]

                if prediction[0] == 1:
                    st.success(f"**Resultado:** Passageiro Provavelmente **SOBREVIVERIA** (Probabilidade: {survival_probability:.2%})")
                else:
                    st.error(f"**Resultado:** Passageiro Provavelmente **NÃO SOBREVIVERIA** (Probabilidade de Sobrevivência: {survival_probability:.2%})")
        else:
            st.error("Não foi possível carregar os artefatos necessários (modelo/pipeline/métricas). Por favor, treine os modelos primeiro.")

    elif page == "Gerar Submissão Kaggle":
        st.title("📥 Gerar Arquivo de Submissão para o Kaggle")
        st.write("Esta seção processa o arquivo `test.csv`, gera as predições com o modelo final e cria o arquivo `submission.csv`.")

        test_file_path = str(settings.TEST_FILE)
        if not os.path.exists(test_file_path):
             st.error(f"Arquivo de teste não encontrado em: `{test_file_path}`. Verifique o caminho em `src/config/settings.py`.")
        elif not model or not fe_pipeline:
            st.error("Não foi possível carregar os artefatos necessários (modelo/pipeline). Por favor, treine os modelos primeiro.")
        else:
            if st.button("Gerar Arquivo", type="secondary", help="Processa o arquivo de teste e gera o arquivo de submissão para o Kaggle."):
                with st.spinner("Carregando e processando os dados de teste..."):
                    try:
                        test_df = pd.read_csv(test_file_path)
                        passenger_ids = test_df['PassengerId']

                        processed_test_df = fe_pipeline.transform(test_df)
                        predictions = model.predict(processed_test_df)

                        submission_df = pd.DataFrame({
                            "PassengerId": passenger_ids,
                            "Survived": predictions
                        })

                        st.success("Arquivo de submissão gerado com sucesso!")
                        st.dataframe(
                            submission_df,
                            use_container_width=True,
                            hide_index=False,
                            height=400
                        )

                        csv_data = submission_df.to_csv(index=False).encode('utf-8')

                        st.download_button(
                           label="📥 Baixar submission.csv",
                           data=csv_data,
                           file_name='submission.csv',
                           mime='text/csv',
                        )

                    except Exception as e:
                        st.error(f"Ocorreu um erro inesperado durante a geração do arquivo: {e}")

    with st.sidebar.expander("⚠️ Opções Avançadas"):
        if st.button("Apagar Modelos", type="primary", help="Remove todos os modelos treinados e volta para a tela de configuração.", use_container_width=True):
            st.session_state['confirm_delete_artifacts'] = True

        if 'confirm_delete_artifacts' in st.session_state and st.session_state['confirm_delete_artifacts']:
            st.warning("Tem certeza que deseja apagar todos os modelos e artefatos? Esta ação não pode ser desfeita.")

            col1, col2 = st.columns(2)
            
            confirm_delete = col1.button(
                "Sim, apagar", 
                key="confirm_delete", 
                use_container_width=True
            )
            cancel_delete = col2.button(
                "Cancelar", 
                key="cancel_delete", 
                use_container_width=True
            )
            
            if confirm_delete:
                delete_artifacts()
                st.success("Modelos apagados. Recarregando...")
                del st.session_state['confirm_delete_artifacts']
                time.sleep(2)
                st.rerun()
            elif cancel_delete:
                del st.session_state['confirm_delete_artifacts']
                st.rerun()
