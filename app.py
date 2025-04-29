from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import os
import glob
from clasificar_imagen_colon import clasificar_imagen_colon

# Inicializar la app
app = FastAPI()

# Permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar las imágenes en la ruta /static
app.mount("/static", StaticFiles(directory="colon_image_sets"), name="static")

# Cargar modelos y datasets
modelo_supervivencia = joblib.load("modelos_guardados/modelo_rf_v2.pkl")
df_pacientes = pd.read_csv("data/historial_medico.csv")
df_analisis = pd.read_csv("data/analisis_cancer.csv")
df_imagenes = pd.read_csv("data/historial_medico_imagenes.csv")

columnas_modelo = [
    "Sexo", "Family history", "smoke", "alcohol", "obesity",
    "diet", "Screening_History", "Healthcare_Access", "tumor_size",
    "relapse", "early_detection", "inflammatory_bowel_disease"
]

def buscar_imagen(imagename):
    # Asumimos que si empieza por "colonca" es adenocarcinoma (1), si es "colonn" es normal (0)
    if "colonca" in imagename:
        ruta = os.path.join("colon_image_sets/1_adenocarcinoma", imagename)
    elif "colonn" in imagename:
        ruta = os.path.join("colon_image_sets/0_normal", imagename)
    else:
        return None  # Si no se reconoce el patrón, no devuelve ruta

    if os.path.isfile(ruta):
        return ruta
    else:
        return None

# ===========================
# RUTA PARA ANALIZAR PACIENTE
# ===========================
@app.get("/analizar_paciente")
def analizar_paciente(id: int):
    if id not in df_pacientes["id"].values:
        return {"error": "El ID introducido no corresponde a ningún paciente."}
    print("La id introducida es:",id)
    paciente = df_pacientes[df_pacientes["id"] == id].copy()
    analisis = df_analisis[df_analisis["id"] == id].copy()
    imagen_row = df_imagenes[df_imagenes["id"] == id]

    if analisis.empty or imagen_row.empty:
        return {"error": "No se ha encontrado información clínica o imagen asociada para este paciente."}

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
    pred_supervivencia = "alta" if prob_supervivencia >= 0.5 else "baja"
    imagename = imagen_row["imagename"].values[0]+".jpeg"

    imagen_path = buscar_imagen(imagename)

    if not imagen_path:
        return {"error": "Imagen no encontrada para el paciente."}

    # Generar la URL pública para la imagen
    relative_path = imagen_path.replace("colon_image_sets/", "")
    imagen_url = f"/static/{relative_path}"

    prob_cancer = clasificar_imagen_colon(imagen_path)
    combined_score = (prob_cancer * (1 - prob_supervivencia)) * 100

    if combined_score < 25:
        riesgo = "bajo"
    elif combined_score < 60:
        riesgo = "moderado"
    else:
        riesgo = "alto"

    return {
        "supervivencia": round(prob_supervivencia * 100, 2),
        "cancer": round(prob_cancer * 100, 2),
        "riesgo_combinado": round(combined_score, 2),
        "riesgo": riesgo,
        "imagen_url": imagen_url  # 🟢 Aquí va la URL de la imagen
    }
