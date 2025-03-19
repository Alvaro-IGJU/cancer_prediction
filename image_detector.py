import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def predecir_imagen(ruta_imagen, modelo, img_size=(224, 224)):
    # Cargar y redimensionar la imagen
    img = image.load_img(ruta_imagen, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    img_array = img_array / 255.0

    # La salida es la probabilidad de la clase “normal” (colon_n)
    prob_normal = modelo.predict(img_array)[0][0]

    # Calculamos probabilidad de cáncer
    prob_cancer = 1 - prob_normal

    # Decidimos la clase según el umbral 0.5
    if prob_normal >= 0.5:
        return f"colon_n (normal) con prob_normal={prob_normal:.4f}"
    else:
        return f"colon_aca (cáncer) con prob_cancer={prob_cancer:.4f}"

# Cargar tu modelo .h5
model = tf.keras.models.load_model('modelo_cancer_colon_vgg16.h5')

# Ejemplo de uso
test_image_path = "colon_image_sets/colon_aca/colonca400.jpeg"
test_image_path = "colon_image_sets/colon_n/colonn60.jpeg"
resultado = predecir_imagen(test_image_path, model)
print(f"Imagen: {test_image_path}")
print("Predicción:", resultado)
