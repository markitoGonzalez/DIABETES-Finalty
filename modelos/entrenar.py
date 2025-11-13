import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Cargando dataset limpio...")
df = pd.read_csv("datasets/cdc_limpio.csv")

# Variables seleccionadas
variables = [
    "HighBP", "HighChol", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "DiffWalk",
    "Sex", "Age", "GenHlth"
]

X = df[variables]
y = df["Diabetes_binary"]

# División entrenamiento / prueba
print("Dividiendo datos...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado SOLO a BMI y Age
print("📏 Escalando variables BMI y Age...")

scaler = StandardScaler()
cols_to_scale = ["BMI", "Age"]

# Crear copias
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Entrenar scaler solo en TRAIN
scaler.fit(X_train[cols_to_scale])

# Aplicar escalado
X_train_scaled[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# Guardar scaler
joblib.dump(scaler, "modelos/scaler_cdc.sav")

# MODELO BASE — REGRESIÓN LOGÍSTICA
print("\nEntrenando Regresión Logística...")

modelo_rl = LogisticRegression(max_iter=300, class_weight="balanced")
modelo_rl.fit(X_train_scaled, y_train)
y_pred_rl = modelo_rl.predict(X_test_scaled)

print("\nRESULTADOS RL:")
print("Accuracy:", accuracy_score(y_test, y_pred_rl))
print(classification_report(y_test, y_pred_rl))
print("Confusion:\n", confusion_matrix(y_test, y_pred_rl))

joblib.dump(modelo_rl, "modelos/modelo_rl_cdc.sav")

# MODELO FINAL — RANDOM FOREST
print("\nEntrenando Random Forest...")

modelo_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

modelo_rf.fit(X_train, y_train)
y_pred_rf = modelo_rf.predict(X_test)

print("\nRESULTADOS RF:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Confusion:\n", confusion_matrix(y_test, y_pred_rf))

joblib.dump(modelo_rf, "modelos/modelo_rf_cdc.sav")

print("\nENTRENAMIENTO COMPLETADO")
print("Modelos guardados en carpeta /modelos")
