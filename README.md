
# 🧬 Sistema de Predicción de Cáncer de Colon

Este proyecto permite predecir la probabilidad de supervivencia de pacientes con cáncer de colon y detectar tumores malignos a partir de imágenes histológicas, utilizando modelos de aprendizaje automático.

---

## 📦 REQUISITOS PREVIOS

### 📁 Descargar el dataset de imágenes

Descargar únicamente el dataset de **colon** desde Kaggle:  
🔗 https://www.kaggle.com/code/mohamedibrahim206/colon-classification-cnn-acc/input

Cambiar los nombres de los directorios descargados:

- `colon_n` → `0_normal`  
- `colon_aca` → `1_adenocarcinoma`

### 🧠 Descargar los modelos `.pkl`

Desde Google Drive:  
🔗 https://drive.google.com/drive/u/0/folders/1RFfbWz8sfqC_Nd9tWqdkE2gCl4gHL_sV

Archivos necesarios:

- `modelo_rf_v2.pkl`  
- `random_forest_colon.pkl`

Guardar los modelos en el directorio `modelos_guardados/`  
> *Si no existe, créalo manualmente.*

---

## 🚀 PARA USAR EL PROGRAMA

```bash
pip install -r requirements.txt
python create_db.py
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Luego, abre `index.html` con la extensión **Open Live Server** de VSCode.

---

## 🧪 PARA ENTRENAR LOS MODELOS

```bash
pip install -r requirements_bigdata.txt
python create_db.py
```

Ejecutar los siguientes notebooks:

- `modelo_prediccion_cancer.ipynb`  
- `modelo_prediccion_imagenes.ipynb`
