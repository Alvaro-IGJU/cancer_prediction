import streamlit as st
import pandas as pd
import joblib
from clasificar_imagen_colon import clasificar_imagen_colon  # asegúrate de tener este archivo correctamente

# Cargar el modelo clínico de supervivencia
modelo = joblib.load("modelos_guardados/modelo_rf_v2.pkl")

st.title("🔬 Predicción combinada de Supervivencia y Probabilidad de Cáncer")
st.markdown("Introduce los datos clínicos y una imagen histológica del tejido de colon para estimar la situación del paciente.")

# --- Formulario de datos clínicos ---
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

# Crear DataFrame con los datos clínicos
datos_paciente = pd.DataFrame([{
    "Sexo": sexo,
    "tumor_size": tumor_size,
    "relapse": relapse,
    "early_detection": early_detection,
    "inflammatory_bowel_disease": bowel,
    "obesity": obesity,
    "Family history": family_history
}])

# --- Subida de imagen histológica ---
st.markdown("### 📷 Imagen histológica del tejido")
uploaded_file = st.file_uploader("Sube una imagen del tejido del colon (JPG o PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen cargada", use_column_width=True)

# --- Botón para lanzar predicción combinada ---
st.markdown("---")
if st.button("🔍 Predecir"):
    # 1. Predicción de supervivencia
    prob_supervivencia = modelo.predict_proba(datos_paciente)[0][1]
    pred_supervivencia = "alta" if prob_supervivencia >= 0.5 else "baja"

    st.subheader("🧬 Predicción clínica de supervivencia:")
    st.markdown(f"**Probabilidad de supervivencia:** {prob_supervivencia * 100:.1f}% ({pred_supervivencia})")

    # 2. Análisis de imagen
    if uploaded_file is not None:
        # Guardar imagen temporalmente
        with open("temp_img.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Obtener probabilidad de cáncer desde la imagen
        prob_cancer = clasificar_imagen_colon("temp_img.jpg")

        st.subheader("🧫 Análisis histológico:")
        st.markdown(f"**Probabilidad de cáncer en imagen:** {prob_cancer * 100:.1f}%")

        # 3. Índice combinado
        combined_score = (prob_cancer * (1 - prob_supervivencia)) * 100
        st.markdown("### 🧮 Índice combinado de riesgo")
        st.markdown(f"**Riesgo combinado estimado:** {combined_score:.1f} (↑ más alto = más riesgo)")

        # Semáforo visual
        if combined_score < 25:
            st.success("🟢 Riesgo bajo")
        elif combined_score < 60:
            st.warning("🟡 Riesgo moderado")
        else:
            st.error("🔴 Riesgo alto")

    else:
        st.warning("⚠️ Sube una imagen para completar el análisis.")
