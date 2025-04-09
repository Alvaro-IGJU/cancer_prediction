import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado
modelo = joblib.load("modelo_rf_v2.pkl")

st.title("🔬 Predicción de Supervivencia en Pacientes con Cáncer")
st.markdown("Completa los siguientes datos para obtener una predicción basada en variables clínicas clave.")

# Crear columnas para el formulario
col1, col2 = st.columns(2)

with col1:
    sexo = st.selectbox("Sexo", ["F", "M"])
    relapse = st.selectbox("¿Ha tenido recaída?", ["No", "Yes"])
    early_detection = st.selectbox("¿Se ha detectado precozmente?", ["No", "Yes"])
    bowel = st.selectbox("¿Tiene enfermedad inflamatoria intestinal?", ["No", "Yes"])

with col2:
    tumor_size = st.number_input("Tamaño del tumor (en mm)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    obesity = st.selectbox("Estado de obesidad", ["Normal", "Overweight", "Obese"])
    family_history = st.selectbox("¿Antecedentes familiares de cáncer?", ["No", "Yes"])

# Crear DataFrame con los datos introducidos
datos_paciente = pd.DataFrame([{
    "Sexo": sexo,
    "tumor_size": tumor_size,
    "relapse": relapse,
    "early_detection": early_detection,
    "inflammatory_bowel_disease": bowel,
    "obesity": obesity,
    "Family history": family_history
}])

# Mostrar los resultados
st.markdown("---")
if st.button("🔍 Predecir supervivencia"):
    pred = modelo.predict(datos_paciente)[0]
    prob = modelo.predict_proba(datos_paciente)[0][1]

    if pred == "Yes":
        st.success(f"✅ Supervivencia probable ({prob * 100:.1f}% de supervivencia)")
    else:
        st.error(f"⚠️ Riesgo de no supervivencia ({(1 - prob) * 100:.1f}% de no supervivencia)")
