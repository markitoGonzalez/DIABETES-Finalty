from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import logging
import os
import base64

app = Flask(__name__)
CORS(app)

# ================================
# LOGGING PARA MLOps
# ================================
logging.basicConfig(
    filename="logs_api.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Iniciando API con el modelo de Regresión Logística...")

# ================================
# RUTAS DE MODELO Y SCALER
# ================================
os.makedirs("modelos", exist_ok=True)  # Asegura que exista la carpeta

ruta_modelo = "modelos/modelo_rl_cdc_v1.sav"
ruta_scaler = "modelos/scaler_cdc.sav"

# ================================
# RECONSTRUIR MODELOS DESDE VARIABLES (RAILWAY)
# ================================
scaler_b64 = os.getenv("SCALER_B64")
model_b64 = os.getenv("MODEL_B64")

if scaler_b64 and not os.path.exists(ruta_scaler):
    try:
        with open(ruta_scaler, "wb") as f:
            f.write(base64.b64decode(scaler_b64))
        logging.info("Scaler reconstruido correctamente desde SCALER_B64.")
    except Exception as e:
        logging.error(f"Error reconstruyendo scaler: {str(e)}")

if model_b64 and not os.path.exists(ruta_modelo):
    try:
        with open(ruta_modelo, "wb") as f:
            f.write(base64.b64decode(model_b64))
        logging.info("Modelo RL reconstruido correctamente desde MODEL_B64.")
    except Exception as e:
        logging.error(f"Error reconstruyendo modelo: {str(e)}")

# ================================
# CARGAR MODELO Y SCALER
# ================================
if not os.path.exists(ruta_modelo):
    raise FileNotFoundError(f"No se encuentra el modelo en {ruta_modelo}")

if not os.path.exists(ruta_scaler):
    raise FileNotFoundError(f"No se encuentra el scaler en {ruta_scaler}")

modelo = joblib.load(ruta_modelo)
scaler = joblib.load(ruta_scaler)

variables_orden = [
    "HighBP", "HighChol", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "DiffWalk",
    "Sex", "Age", "GenHlth"
]

cols_to_scale = ["BMI", "Age"]

# ================================
# ENDPOINT DE PREDICCIÓN
# ================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validar todas las variables
        for v in variables_orden:
            if v not in data:
                return jsonify({"error": f"Falta la variable: {v}"}), 400

            # Deben ser numéricas
            if not isinstance(data[v], (int, float)):
                return jsonify({"error": f"La variable {v} debe ser numérica"}), 400

        # Crear DataFrame
        df = pd.DataFrame([[data[v] for v in variables_orden]], columns=variables_orden)

        # Aplicar scaler a BMI y Age
        df_scaled = df.copy()
        df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])

        # Predicción
        pred = int(modelo.predict(df_scaled)[0])
        prob = float(modelo.predict_proba(df_scaled)[0][1])

        # Logging
        logging.info(f"Entrada: {data} -> Pred: {pred}, Prob: {prob:.4f}")

        return jsonify({
            "resultado": pred,
            "probabilidad": prob
        })

    except Exception as e:
        logging.error(f"ERROR en /predict: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK", "modelo": "LogisticRegression_v1"})


@app.route("/", methods=["GET"])
def home():
    return "API funcionando correctamente – Modelo: Regresión Logística"


if __name__ == "__main__":
    app.run(debug=True)
