from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)


#RGAR MODELO Y SCALER
modelo = joblib.load("modelos/modelo_rf_cdc.sav")     # Modelo final
scaler = joblib.load("modelos/scaler_cdc.sav")        # Scaler BMI y Age

# Variables EXACTAS usadas en el entrenamiento
variables_orden = [
    "HighBP", "HighChol", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "DiffWalk",
    "Sex", "Age", "GenHlth"
]

#ENDPOINT DE PREDICCIÓN
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        # Validar todas las variables
        for v in variables_orden:
            if v not in data:
                return jsonify({"error": f"Falta la variable: {v}"}), 400

        # Crear DataFrame con el orden correcto
        df = pd.DataFrame([[data[v] for v in variables_orden]], columns=variables_orden)

        # PREDICCIÓN
        pred = int(modelo.predict(df)[0])
        prob = float(modelo.predict_proba(df)[0][1])

        return jsonify({
            "resultado": pred,
            "probabilidad": prob
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "API funcionando correctamente"


if __name__ == "__main__":
    app.run(debug=True)
