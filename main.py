import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Cargar el modelo con pipeline (preprocesa internamente)
modelo = joblib.load("modelo_rf_ohe.pkl")

st.title("Predicción de Supervivencia en Pacientes con Cáncer")
st.header("Introduce los datos del paciente")

# Formulario con las variables necesarias
sexo = st.selectbox("Sexo", ["F", "M"])
age = st.slider("Edad", 18, 100, 50)
family_history = st.selectbox("Antecedentes familiares", ["No", "Yes"])
smoke = st.selectbox("Fuma", ["No", "Yes"])
alcohol = st.selectbox("Bebe alcohol", ["No", "Yes"])
obesity = st.selectbox("Obesidad", ["Normal", "Overweight", "Obese"])
diet = st.selectbox("Dieta", ["Low", "Moderate", "High"])
screening = st.selectbox("Historial de cribado", ["Never", "Irregular", "Regular"])
access = st.selectbox("Acceso a la sanidad", ["Low", "Moderate", "High"])
cancer_stage = st.selectbox("Estadio del cáncer", [1, 2, 3, 4])
tumor_size = st.slider("Tamaño del tumor", 0.0, 10.0, 5.0)
early_detection = st.selectbox("Detección precoz", ["No", "Yes"])
bowel = st.selectbox("Enfermedad inflamatoria intestinal", ["No", "Yes"])
relapse = st.selectbox("Ha tenido recaída", ["No", "Yes"])
hemoglobina = st.number_input("Hemoglobina", value=13.0)

# Crear DataFrame con los datos introducidos (sin codificar)
datos_paciente = pd.DataFrame([{
    "Sexo": sexo,
    "Age": age,
    "Family history": family_history,
    "smoke": smoke,
    "alcohol": alcohol,
    "obesity": obesity,
    "diet": diet,
    "Screening_History": screening,
    "Healthcare_Access": access,
    "cancer_stage": cancer_stage,
    "tumor_size": tumor_size,
    "early_detection": early_detection,
    "inflammatory_bowel_disease": bowel,
    "relapse": relapse,
    "Hemoglobina": hemoglobina
}])

# Predicción
if st.button("Predecir supervivencia"):
    pred = modelo.predict(datos_paciente)[0]
    prob = modelo.predict_proba(datos_paciente)[0][1]

    if pred == 1:
        st.success(f"✅ Supervivencia probable ({prob*100:.1f}% de supervivencia)")
    else:
        st.warning(f"⚠️ Riesgo de no supervivencia ({(1 - prob)*100:.1f}% de no supervivencia)")

