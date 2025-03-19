import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

# ---------------------------------------
# 1. Parámetros y rutas
# ---------------------------------------
DATA_DIR = 'colon_image_sets'         # carpeta raíz con subcarpetas: colon_aca, colon_n
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4  # un poco más bajo que 1e-3

# ---------------------------------------
# 2. Generadores de imágenes
# ---------------------------------------
# Separamos 80% para entrenamiento y 20% para validación
# Ajustamos la augmentación para que no sea demasiado agresiva
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,      # rotación moderada
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=42
)

print("\nAsignación de clases:", train_generator.class_indices)
print("Imágenes en entrenamiento:", train_generator.n)
print("Imágenes en validación:", val_generator.n)

# ---------------------------------------
# 3. Modelo: Transfer Learning con VGG16
# ---------------------------------------
# Cargamos VGG16 pre-entrenado en ImageNet, sin las capas fully-connected finales
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
# Congelamos sus capas para preservar los pesos pre-entrenados
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# ---------------------------------------
# 4. Compilar y entrenar
# ---------------------------------------
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# ---------------------------------------
# 5. Evaluación
# ---------------------------------------
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"\nPrecisión en validación: {val_accuracy:.4f} - Pérdida en validación: {val_loss:.4f}")

# ---------------------------------------
# 6. Predicciones y matriz de confusión
# ---------------------------------------
y_pred_proba = model.predict(val_generator)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
y_true = val_generator.classes

cm = confusion_matrix(y_true, y_pred)
print("\nMatriz de confusión (filas = etiqueta verdadera, columnas = predicción):")
print(cm)

# ---------------------------------------
# 7. Guardar gráficos en vez de mostrarlos
# ---------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(acc, label='Entrenamiento')
plt.plot(val_acc, label='Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(loss, label='Entrenamiento')
plt.plot(val_loss, label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.savefig('training_plots.png')  # Guardamos la figura
plt.close()                       # Cerramos para no mostrar

# Heatmap de la matriz de confusión
import seaborn as sns
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(train_generator.class_indices.keys()),
            yticklabels=list(train_generator.class_indices.keys()))
plt.title('Matriz de confusión')
plt.xlabel('Etiqueta predicha')
plt.ylabel('Etiqueta verdadera')
plt.savefig('confusion_matrix.png')
plt.close()

# ---------------------------------------
# 8. Guardar el modelo
# ---------------------------------------
model.save('modelo_cancer_colon_vgg16.h5')
print("\nModelo guardado en 'modelo_cancer_colon_vgg16.h5'")

# ---------------------------------------
# 9. Predicción en una sola imagen
# ---------------------------------------
# Ajusta la ruta a una imagen de prueba (por ejemplo, una imagen en 'colon_n')
test_image_path = 'colon_image_sets/colon_n/colonn1.jpeg'
if os.path.exists(test_image_path):
    img = load_img(test_image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    img_array /= 255.0

    prob = model.predict(img_array)[0][0]
    clase_pred = "colon_aca (cáncer)" if prob < 0.5 else "colon_n (normal)"
    print(f"\nPredicción para '{test_image_path}': {clase_pred} con prob = {prob:.4f}")
else:
    print(f"\nNo se encontró la imagen de prueba: {test_image_path}")
