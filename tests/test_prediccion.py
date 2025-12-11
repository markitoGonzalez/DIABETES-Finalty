import joblib
import pandas as pd
import numpy as np
import os

# Ruta de archivos
ruta_modelo = "modelos/modelo_rl_cdc_v1.sav"
ruta_scaler = "modelos/scaler_cdc.sav"

def test_archivos_existen():
    """Verifica que el modelo y el scaler existan."""
    assert os.path.exists(ruta_modelo), "❌ No se encontró el modelo entrenado."
    assert os.path.exists(ruta_scaler), "❌ No se encontró el scaler."


def test_prediccion_basica():
    """Prueba una predicción usando valores válidos."""
    
    # Cargar modelo y scaler
    modelo = joblib.load(ruta_modelo)
    scaler = joblib.load(ruta_scaler)

    variables_orden = [
        "HighBP", "HighChol", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "DiffWalk",
        "Sex", "Age", "GenHlth"
    ]

    cols_to_scale = ["BMI", "Age"]

    # Ejemplo de entrada
    entrada = {
        "HighBP": 1,
        "HighChol": 0,
        "BMI": 28,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "DiffWalk": 0,
        "Sex": 1,
        "Age": 6,
        "GenHlth": 3
    }

    # Crear DataFrame y escalar
    df = pd.DataFrame([entrada], columns=variables_orden)
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Realizar predicción
    pred = modelo.predict(df_scaled)[0]

    # Asegurar que la predicción sea válida
    assert pred in [0, 1], "❌ La predicción no es válida (debe ser 0 o 1)."


def test_probabilidad_valida():
    """Prueba que la probabilidad de salida esté entre 0 y 1."""

    modelo = joblib.load(ruta_modelo)
    scaler = joblib.load(ruta_scaler)

    entrada = {
        "HighBP": 1,
        "HighChol": 1,
        "BMI": 30,
        "Smoker": 1,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "DiffWalk": 0,
        "Sex": 0,
        "Age": 7,
        "GenHlth": 4
    }

    vars_orden = [
        "HighBP", "HighChol", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "DiffWalk",
        "Sex", "Age", "GenHlth"
    ]

    cols_to_scale = ["BMI", "Age"]

    df = pd.DataFrame([entrada], columns=vars_orden)
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])

    prob = modelo.predict_proba(df_scaled)[0][1]

    assert 0 <= prob <= 1, "❌ La probabilidad está fuera del rango [0, 1]."
