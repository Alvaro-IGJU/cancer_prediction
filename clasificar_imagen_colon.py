import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Cargar el modelo base de extracción de características
_base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, pooling='avg', weights='imagenet')
_feature_model = tf.keras.Model(inputs=_base_model.input, outputs=_base_model.output)

def clasificar_imagen_colon(ruta_imagen):
    """
    Devuelve la probabilidad de que la imagen sea adenocarcinoma (cáncer).

    Args:
        ruta_imagen (str): Ruta a la imagen a clasificar.

    Returns:
        probabilidad_cancer (float): Probabilidad de que sea cáncer (clase 0) entre 0 y 1
    """
    # Cargar y preprocesar la imagen
    img = load_img(ruta_imagen, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extraer features
    features = _feature_model.predict(img_array)

    # Cargar el modelo Random Forest
    clf = joblib.load('modelos_guardados/random_forest_colon.pkl')

    # Predecir probabilidades
    probas = clf.predict_proba(features)[0]
    print(f"[DEBUG] Probabilidades predict_proba: {probas}")  # Para inspección

    # Clase 0 = cáncer (adenocarcinoma)
    prob_cancer = float(probas[1])
    return prob_cancer
