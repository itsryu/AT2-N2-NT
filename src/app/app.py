import streamlit as st
import pandas as pd
import joblib
import os
from typing import Any

st.set_page_config(
    page_title="Previsão de Sobrevivência - Titanic",
    page_icon="🚢",
    layout="wide"
)

@st.cache_resource
def load_model(path: str) -> Any:
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Arquivo do modelo não encontrado em '{path}'. Por favor, execute o script de treinamento primeiro.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")
        return None


MODEL_PATH = '../../artifacts/best_model.joblib'
model = load_model(MODEL_PATH)


st.title("🚢 Previsão de Sobrevivência no Titanic")
st.markdown("""
Esta aplicação utiliza um modelo de Machine Learning para prever a probabilidade de um passageiro sobreviver ao desastre do Titanic.
Insira as informações do passageiro na barra lateral à esquerda para obter uma previsão.
""")


st.sidebar.header("Informações do Passageiro")

def user_input_features() -> pd.DataFrame:
    pclass = st.sidebar.selectbox("Classe da Passagem (Pclass)", (1, 2, 3))
    sex = st.sidebar.selectbox("Sexo", ("Masculino", "Feminino"))
    age = st.sidebar.slider("Idade (Age)", 0, 100, 29)
    sibsp = st.sidebar.slider("Nº de Irmãos/Cônjuges a Bordo (SibSp)", 0, 8, 0)
    parch = st.sidebar.slider("Nº de Pais/Filhos a Bordo (Parch)", 0, 6, 0)
    fare = st.sidebar.number_input("Tarifa (Fare)", 0.0, 600.0, 32.2)
    embarked = st.sidebar.selectbox("Porto de Embarque (Embarked)", ("Southampton", "Cherbourg", "Queenstown"))
    
    sex_map = {"Masculino": 0, "Feminino": 1}
    embarked_map = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}
    
    data = {
        'Pclass': pclass,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Sex': sex_map[sex],
        'Embarked': embarked_map[embarked]
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

if model:
    input_df = user_input_features()
    
    st.subheader("Dados do Passageiro Inseridos:")
    st.write(input_df)
    
    if st.button("Realizar Previsão", type="primary"):
        try:
            prediction_proba = model.predict_proba(input_df)
            prediction = model.predict(input_df)

            st.subheader("Resultado da Previsão")
            
            survival_probability = prediction_proba[0][1]

            col1, col2 = st.columns([1, 2])
            
            with col1:
                if prediction[0] == 1:
                    st.image("https://em-content.zobj.net/source/microsoft-teams/363/thumbs-up_1f44d.png", width=100)
                else:
                    st.image("https://em-content.zobj.net/source/microsoft-teams/363/thumbs-down_1f44e.png", width=100)

            with col2:
                if prediction[0] == 1:
                    st.success(f"**Provavelmente Sobreviveria**")
                    st.progress(survival_probability)
                    st.metric(label="Probabilidade de Sobrevivência", value=f"{survival_probability:.2%}")
                else:
                    st.error(f"**Provavelmente NÃO Sobreviveria**")
                    st.progress(survival_probability)
                    st.metric(label="Probabilidade de Sobrevivência", value=f"{survival_probability:.2%}")
        except Exception as e:
            st.error(f"Ocorreu um erro durante a predição: {e}")

else:
    st.warning("O modelo não está carregado. Execute o script `src/models/train_model.py` para treinar e salvar um modelo.")
