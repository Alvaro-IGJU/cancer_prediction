# predecir.py
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image

# Cargamos el modelo solo una vez
model = tf.keras.models.load_model('modelo_cancer_colon_vgg16.h5')

def predecir_imagen_bytes(image_bytes, img_size=(224, 224)):
    """
    Recibe la imagen en formato bytes, la procesa y obtiene la predicción
    usando el modelo cargado. Devuelve un string con la clase y la prob. de cáncer.
    """
    # Abrir la imagen desde bytes y redimensionar
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image = image.resize(img_size)

    # Convertir a array y escalar
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    img_array /= 255.0

    # El modelo predice la prob. de "normal"
    prob_normal = model.predict(img_array)[0][0]

    # Probabilidad de cáncer en %
    prob_cancer_percent = (1 - prob_normal) * 100

    # Si la prob. de cáncer >= 50%, clasificamos como cáncer
    if prob_cancer_percent >= 50:
        return f"Tumor maligno<br>Probabilidad de cáncer: {prob_cancer_percent:.2f}%"
    else:
        return f"Tumor benigno<br>Probabilidad de cáncer: {prob_cancer_percent:.2f}%"
