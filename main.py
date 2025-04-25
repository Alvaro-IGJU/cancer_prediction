import streamlit as st
import pandas as pd
import joblib
import os
from clasificar_imagen_colon import clasificar_imagen_colon  # asegúrate de tener esta función correctamente definida
import glob

# Cargar datasets
df_pacientes = pd.read_csv("data/historial_medico.csv")
df_analisis = pd.read_csv("data/analisis_cancer.csv")
df_imagenes = pd.read_csv("data/historial_medico_imagenes.csv")

# Cargar modelo de supervivencia
modelo_supervivencia = joblib.load("modelos_guardados/modelo_rf_v2.pkl")

# Función para buscar la ruta real de la imagen

def buscar_imagen(imagename):
    base_id = os.path.splitext(imagename)[0]  # sin extensión
    posibles_rutas = glob.glob(f"colon_image_sets/0_normal/*{base_id}*.jpeg") + \
                     glob.glob(f"colon_image_sets/1_adenocarcinoma/*{base_id}*.jpeg")

    for ruta in posibles_rutas:
        st.text(f"🔍 Imagen encontrada: {ruta}")
        if os.path.isfile(ruta):
            return ruta

    st.text("⚠️ No se encontró ninguna imagen que coincida con ese nombre.")
    return None




# Interfaz
st.title("🔬 Análisis de paciente mediante ID")
st.markdown("Introduce el ID del paciente para obtener la predicción combinada de supervivencia y probabilidad de cáncer.")

# Introducir ID
id_input = st.number_input("ID del paciente", min_value=1, step=1)

# Botón para lanzar predicción
if st.button("🔍 Analizar paciente"):
    if id_input in df_pacientes["id"].values:
        paciente = df_pacientes[df_pacientes["id"] == id_input].copy()
        analisis = df_analisis[df_analisis["id"] == id_input].copy()
        imagen_row = df_imagenes[df_imagenes["id"] == id_input]

        if not analisis.empty:
            input_modelo = pd.DataFrame([{
                "Sexo": paciente["Sexo"].values[0],
                "tumor_size": analisis["tumor_size"].values[0],
                "relapse": analisis["relapse"].values[0],
                "early_detection": analisis["early_detection"].values[0],
                "inflammatory_bowel_disease": analisis["inflammatory_bowel_disease"].values[0],
                "obesity": paciente["obesity"].values[0],
                "Family history": paciente["Family history"].values[0]
            }])

            prob_supervivencia = modelo_supervivencia.predict_proba(input_modelo)[0][1]
            pred_supervivencia = "alta" if prob_supervivencia >= 0.5 else "baja"

            st.subheader("🧬 Predicción clínica de supervivencia:")
            st.markdown(f"**Probabilidad de supervivencia:** {prob_supervivencia * 100:.1f}% ({pred_supervivencia})")

            if not imagen_row.empty:
                imagename = imagen_row["imagename"].values[0]
                imagen_path = buscar_imagen(imagename)

                if imagen_path:
                    st.subheader("🧫 Análisis histológico:")
                    st.image(imagen_path, caption="Imagen asociada al paciente", use_column_width=True)

                    prob_cancer = clasificar_imagen_colon(imagen_path)
                    st.markdown(f"**Probabilidad de cáncer en imagen:** {prob_cancer * 100:.1f}%")

                    combined_score = (prob_cancer * (1 - prob_supervivencia)) * 100
                    st.markdown("### 🧮 Índice combinado de riesgo")
                    st.markdown(f"**Riesgo combinado estimado:** {combined_score:.1f} (↑ más alto = más riesgo)")

                    if combined_score < 25:
                        st.success("🟢 Riesgo bajo")
                    elif combined_score < 60:
                        st.warning("🟡 Riesgo moderado")
                    else:
                        st.error("🔴 Riesgo alto")
                else:
                    st.warning("⚠️ Imagen no encontrada en las carpetas '0_normal' o '1_adenocarcinoma'.")
            else:
                st.warning("⚠️ No se ha encontrado imagen asociada al paciente.")
        else:
            st.warning("⚠️ No se ha encontrado información oncológica para este paciente.")
    else:
        st.error("❌ El ID introducido no corresponde a ningún paciente.")
