import pandas as pd
import joblib
import os
import glob
import time
from clasificar_imagen_colon import clasificar_imagen_colon

# ================================
# CONFIGURACIÓN Y CARGA DE DATOS
# ================================
modelo_supervivencia = joblib.load("modelos_guardados/modelo_rf_v2.pkl")
df_pacientes = pd.read_csv("data/historial_medico.csv")
df_analisis = pd.read_csv("data/analisis_cancer.csv")
df_imagenes = pd.read_csv("data/historial_medico_imagenes.csv")

columnas_modelo = [
    "Sexo", "Family history", "smoke", "alcohol", "obesity",
    "diet", "Screening_History", "Healthcare_Access", "tumor_size",
    "relapse", "early_detection", "inflammatory_bowel_disease"
]

# ================================
# FUNCIÓN PARA BUSCAR LA IMAGEN
# ================================
def buscar_imagen(imagename):
    base_id = os.path.splitext(imagename)[0]
    posibles_rutas = glob.glob(f"colon_image_sets/0_normal/*{base_id}*.jpeg") + \
                     glob.glob(f"colon_image_sets/1_adenocarcinoma/*{base_id}*.jpeg")
    for ruta in posibles_rutas:
        if os.path.isfile(ruta):
            return ruta
    return None

# ================================
# PRUEBA CON VARIOS PACIENTES
# ================================
def validar_modelo(num_pacientes=20):
    start_time = time.time()
    aciertos = 0
    total = 0

    ids_disponibles = df_pacientes["id"].values[:num_pacientes]  # coge los primeros N pacientes (puedes cambiar esto)

    for paciente_id in ids_disponibles:
        paciente = df_pacientes[df_pacientes["id"] == paciente_id].copy()
        analisis = df_analisis[df_analisis["id"] == paciente_id].copy()
        imagen_row = df_imagenes[df_imagenes["id"] == paciente_id]

        if analisis.empty or imagen_row.empty:
            print(f"⚠️ Paciente ID {paciente_id} sin datos suficientes, saltando...")
            continue

        input_modelo = pd.DataFrame([{
            "Sexo": paciente["Sexo"].values[0],
            "Family history": paciente["Family history"].values[0],
            "smoke": paciente["smoke"].values[0],
            "alcohol": paciente["alcohol"].values[0],
            "obesity": paciente["obesity"].values[0],
            "diet": paciente["diet"].values[0],
            "Screening_History": paciente["Screening_History"].values[0],
            "Healthcare_Access": paciente["Healthcare_Access"].values[0],
            "tumor_size": analisis["tumor_size"].values[0],
            "relapse": analisis["relapse"].values[0],
            "early_detection": analisis["early_detection"].values[0],
            "inflammatory_bowel_disease": analisis["inflammatory_bowel_disease"].values[0]
        }], columns=columnas_modelo)

        prob_supervivencia = modelo_supervivencia.predict_proba(input_modelo)[0][1]
        pred_supervivencia = "NO tiene cáncer" if prob_supervivencia >= 0.5 else "TIENE cáncer"

        imagename = imagen_row["imagename"].values[0]
        imagen_path = buscar_imagen(imagename)

        if not imagen_path:
            print(f"⚠️ Imagen no encontrada para paciente ID {paciente_id}, saltando...")
            continue

        prob_cancer = clasificar_imagen_colon(imagen_path)
        pred_cancer = "TIENE cáncer" if prob_cancer >= 0.5 else "NO tiene cáncer"

        # Acierto si coinciden ambas predicciones
        es_acierto = pred_supervivencia == pred_cancer
        resultado = "✅ Correcto" if es_acierto else "❌ Incorrecto"
        print(f"Paciente ID {paciente_id} → Supervivencia: {pred_supervivencia}, Imagen: {pred_cancer} → {resultado}")

        total += 1
        if es_acierto:
            aciertos += 1

    elapsed_time = time.time() - start_time
    print("\n===============================")
    print(f"Total pacientes analizados: {total}")
    print(f"Aciertos: {aciertos} / {total} → {100 * aciertos / total:.2f}% de acierto")
    print(f"⏱️ Tiempo total: {elapsed_time:.2f} segundos")
    print("===============================")

# ================================
# EJECUTAR LA VALIDACIÓN
# ================================
if __name__ == "__main__":
    validar_modelo(num_pacientes=1000)  # Cambia el número si quieres probar más o menos pacientes
