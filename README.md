# 🧬 Sistema de Predicción de Cáncer de Colon

Este sistema combina análisis clínico y visual mediante inteligencia artificial para predecir la probabilidad de cáncer de colon y la supervivencia del paciente.

---

## ✅ Requisitos previos

### 1. **Descargar el dataset de imágenes**

- Accede al siguiente enlace de Kaggle:  
  👉 [Colon Classification Dataset](https://www.kaggle.com/code/mohamedibrahim206/colon-classification-cnn-acc/input)  
  *(Solo es necesario el dataset de colon)*

- Cambia los nombres de los directorios descargados:
  - `colon_n` → `0_normal`
  - `colon_aca` → `1_adenocarcinoma`

### 2. **Descargar los modelos entrenados (.pkl)**

- Descarga desde Google Drive:  
  👉 [Modelos .pkl](https://drive.google.com/drive/u/0/folders/1RFfbWz8sfqC_Nd9tWqdkE2gCl4gHL_sV)

- Archivos necesarios:
  - `modelo_rf_v2.pkl`
  - `random_forest_colon.pkl`

- Guardar en el directorio `modelos_guardados/`  
  *(Crear la carpeta si no existe)*

---

## 🚀 Iniciar la aplicación

```bash
pip install -r requirements.txt
python create_db.py
uvicorn app:app --reload --host 0.0.0.0 --port 8000
