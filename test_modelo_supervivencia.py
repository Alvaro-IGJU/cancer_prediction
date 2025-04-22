import joblib
import pandas as pd
import numpy as np
import pytest

# ========================
# CONFIGURACIÓN DEL MODELO
# ========================

modelo_path = "modelos_guardados/modelo_rf_v2.pkl"
modelo = joblib.load(modelo_path)

columnas_modelo = [
    "Sexo", "Family history", "smoke", "alcohol", "obesity",
    "diet", "Screening_History", "Healthcare_Access", "tumor_size",
    "relapse", "early_detection", "inflammatory_bowel_disease"
]

input_correcto = pd.DataFrame([{
    "Sexo": "M",
    "Family history": "No",
    "smoke": "No",
    "alcohol": "No",
    "obesity": "Normal",
    "diet": "Moderate",
    "Screening_History": "Regular",
    "Healthcare_Access": "High",
    "tumor_size": 40,
    "relapse": "No",
    "early_detection": "Yes",
    "inflammatory_bowel_disease": "No"
}], columns=columnas_modelo)

# ====================
# PRUEBAS UNITARIAS
# ====================

def test_modelo_carga():
    assert modelo is not None, "El modelo no se ha cargado correctamente"
    print("✅ test_modelo_carga: PASADO")

def test_prediccion_output_formato():
    probas = modelo.predict_proba(input_correcto)
    assert isinstance(probas, np.ndarray), "El output no es del tipo esperado (numpy.ndarray)"
    print("✅ test_prediccion_output_formato: PASADO")

def test_probabilidad_en_rango():
    probas = modelo.predict_proba(input_correcto)[0]
    assert 0 <= probas[0] <= 1 and 0 <= probas[1] <= 1, "La probabilidad está fuera del rango [0, 1]"
    assert abs(probas[0] + probas[1] - 1) < 0.01, "Las probabilidades no suman aproximadamente 1"
    print("✅ test_probabilidad_en_rango: PASADO")

def test_output_shape():
    probas = modelo.predict_proba(input_correcto)
    assert len(probas[0]) == 2, "El modelo debería devolver dos probabilidades (clase 0 y clase 1)"
    print("✅ test_output_shape: PASADO")

def test_error_si_faltan_campos():
    input_incompleto = pd.DataFrame([{
        "Sexo": "M",
        "tumor_size": 40  # Faltan campos
    }])
    try:
        modelo.predict_proba(input_incompleto)
        print("❌ test_error_si_faltan_campos: FALLADO (no lanzó excepción)")
    except Exception:
        print("✅ test_error_si_faltan_campos: PASADO")

# ==========================
# LLAMADA A LOS TESTS AQUÍ
# ==========================

if __name__ == "__main__":
    test_modelo_carga()
    test_prediccion_output_formato()
    test_probabilidad_en_rango()
    test_output_shape()
    test_error_si_faltan_campos()
