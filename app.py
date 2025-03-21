# app.py
from flask import Flask, request, render_template_string
import base64
from image_prediction import predecir_imagen_bytes

app = Flask(__name__)

@app.route('/')
def index():
    # Formulario para subir imagen
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Detector de Cáncer de Colon</title>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    </head>
    <body class="bg-light">
      <div class="container my-4">
        <h1 class="text-center">Detector de Cáncer de Colon</h1>
        <div class="card p-4 shadow-sm">
          <form method="POST" action="/predict" enctype="multipart/form-data">
            <div class="mb-3">
              <label for="image" class="form-label">Sube una imagen:</label>
              <input class="form-control" type="file" name="image" accept="image/*" required>
            </div>
            <button class="btn btn-primary" type="submit">Predecir</button>
          </form>
        </div>
      </div>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/predict', methods=['POST'])
def predict():
    # Verificamos que se haya subido un archivo
    if 'image' not in request.files:
        return "No se envió ninguna imagen"

    file = request.files['image']
    if file.filename == '':
        return "Nombre de archivo inválido"

    # Leer el contenido del archivo en memoria
    image_bytes = file.read()

    # Convertir la imagen a Base64 para mostrarla en la página
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    # Construir el src para la etiqueta <img>
    image_src = f"data:image/jpeg;base64,{image_base64}"

    # Llamar a la función de predicción (en predecir.py)
    resultado = predecir_imagen_bytes(image_bytes)

    # Devolvemos una página que muestra la imagen y la predicción
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <title>Resultado de la Predicción</title>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    </head>
    <body class="bg-light">
      <div class="container my-4">
        <h2>Resultado de la predicción</h2>
        <div class="card p-4 shadow-sm mb-3">
          <img src="{image_src}" alt="Imagen subida" class="img-fluid mb-3" style="max-width:400px;">
          <h5>{resultado}</h5>
        </div>
        <a href="/" class="btn btn-secondary">Volver</a>
      </div>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
