import sqlite3
import pandas as pd
import joblib
import os
import glob
from clasificar_imagen_colon import clasificar_imagen_colon

# ==============================
# CARGA DE MODELO Y CONEXIÓN DB
# ==============================
modelo_supervivencia = joblib.load("modelos_guardados/modelo_rf_v2.pkl")
conn = sqlite3.connect("prediccion_cancer.db")

# Columnas esperadas por el modelo
columnas_modelo = [
    "Sexo", "Family history", "smoke", "alcohol", "obesity",
    "diet", "Screening_History", "Healthcare_Access", "tumor_size",
    "relapse", "early_detection", "inflammatory_bowel_disease"
]

# ==============================
# FUNCIÓN PARA BUSCAR IMAGEN
# ==============================
def buscar_imagen(imagename):
    base_id = os.path.splitext(imagename)[0]
    posibles_rutas = glob.glob(f"colon_image_sets/0_normal/*{base_id}*.jpeg") + \
                     glob.glob(f"colon_image_sets/1_adenocarcinoma/*{base_id}*.jpeg")
    for ruta in posibles_rutas:
        if os.path.isfile(ruta):
            return ruta
    return None

# ==============================
# PRUEBA DE INTEGRACIÓN COMPLETA
# ==============================
def prueba_integracion(num_pacientes=10):
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM pacientes LIMIT ?", (num_pacientes,))
    ids = [row[0] for row in cursor.fetchall()]

    errores = 0

    for paciente_id in ids:
        try:
            # 1. Cargar datos clínicos
            df_paciente = pd.read_sql_query(f"SELECT * FROM pacientes WHERE id = {paciente_id}", conn)
            df_analisis = pd.read_sql_query(f"SELECT * FROM analisis_cancer WHERE id = {paciente_id}", conn)
            df_imagen = pd.read_sql_query(f"SELECT * FROM imagenes WHERE id = {paciente_id}", conn)

            if df_analisis.empty or df_imagen.empty:
                print(f"⚠️ Paciente ID {paciente_id}: faltan datos, saltando...")
                continue

            # 2. Preparar datos para el modelo clínico
            input_modelo = pd.DataFrame([{
                "Sexo": df_paciente["Sexo"].values[0],
                "Family history": df_paciente["Family history"].values[0],
                "smoke": df_paciente["smoke"].values[0],
                "alcohol": df_paciente["alcohol"].values[0],
                "obesity": df_paciente["obesity"].values[0],
                "diet": df_paciente["diet"].values[0],
                "Screening_History": df_paciente["Screening_History"].values[0],
                "Healthcare_Access": df_paciente["Healthcare_Access"].values[0],
                "tumor_size": df_analisis["tumor_size"].values[0],
                "relapse": df_analisis["relapse"].values[0],
                "early_detection": df_analisis["early_detection"].values[0],
                "inflammatory_bowel_disease": df_analisis["inflammatory_bowel_disease"].values[0]
            }], columns=columnas_modelo)

            # 3. Predicción clínica
            prob_supervivencia = modelo_supervivencia.predict_proba(input_modelo)[0][1]
            pred_supervivencia = "NO tiene cáncer" if prob_supervivencia >= 0.5 else "TIENE cáncer"

            # 4. Predicción con imagen
            imagename = df_imagen["imagename"].values[0]
            imagen_path = buscar_imagen(imagename)
            if not imagen_path:
                print(f"⚠️ Imagen no encontrada para paciente ID {paciente_id}, saltando...")
                continue

            prob_cancer = clasificar_imagen_colon(imagen_path)
            pred_cancer = "TIENE cáncer" if prob_cancer >= 0.5 else "NO tiene cáncer"

            # 5. Mostrar resultado combinado
            resultado = "✅ Correcto" if pred_supervivencia == pred_cancer else "❌ Incorrecto"
            print(f"Paciente ID {paciente_id} → Supervivencia: {pred_supervivencia}, Imagen: {pred_cancer} → {resultado}")

        except Exception as e:
            print(f"❌ Error procesando paciente ID {paciente_id}: {e}")
            errores += 1

    print("\n===============================")
    print(f"Pacientes analizados: {num_pacientes}")
    print(f"Errores detectados durante la prueba: {errores}")
    print("===============================")

# ==============================
# EJECUTAR PRUEBA DE INTEGRACIÓN
# ==============================
if __name__ == "__main__":
    prueba_integracion(num_pacientes=10)
