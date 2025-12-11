import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("=============================================")
print(" CARGANDO DATASET LIMPIO DESDE processed/")
print("=============================================")

ruta_dataset = "datasets/processed/cdc_limpio.csv"

if not os.path.exists(ruta_dataset):
    raise FileNotFoundError(f"ERROR: No se encuentra el archivo {ruta_dataset}")

df = pd.read_csv(ruta_dataset)

variables = [
    "HighBP", "HighChol", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "DiffWalk",
    "Sex", "Age", "GenHlth"
]

X = df[variables]
y = df["Diabetes_binary"]

print("\nDividiendo datos (Train: 80% / Test: 20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====================================================
#    REGRESIÓN LOGÍSTICA — PREPROCESAMIENTO
# ====================================================
print("\n=============================================")
print(" PREPROCESAMIENTO PARA REGRESIÓN LOGÍSTICA")
print("=============================================")

scaler = StandardScaler()
cols_to_scale = ["BMI", "Age"]

X_train_lr = X_train.copy()
X_test_lr = X_test.copy()

# entrenar scaler SOLO con TRAIN
scaler.fit(X_train_lr[cols_to_scale])

# aplicar escalado
X_train_lr[cols_to_scale] = scaler.transform(X_train_lr[cols_to_scale])
X_test_lr[cols_to_scale] = scaler.transform(X_test_lr[cols_to_scale])

# guardar scaler
joblib.dump(scaler, "modelos/scaler_cdc.sav")
print("Scaler guardado en modelos/scaler_cdc.sav")


# ====================================================
#    ENTRENAMIENTO REGRESIÓN LOGÍSTICA
# ====================================================
print("\nEntrenando modelo de Regresión Logística...")

modelo_rl = LogisticRegression(max_iter=400, class_weight="balanced")
modelo_rl.fit(X_train_lr, y_train)

y_pred_rl = modelo_rl.predict(X_test_lr)
acc_rl = accuracy_score(y_test, y_pred_rl)

print("\n============== RESULTADOS RL ==============")
print(classification_report(y_test, y_pred_rl))
print(f"Precisión (Accuracy): {acc_rl * 100:.2f}%")

# guardar modelo RL
joblib.dump(modelo_rl, "modelos/modelo_rl_cdc_v1.sav")
print("Modelo RL guardado como modelo_rl_cdc_v1.sav")


# ====================================================
#    ENTRENAMIENTO RANDOM FOREST
# ====================================================
print("\n=============================================")
print(" ENTRENANDO RANDOM FOREST")
print("=============================================")

modelo_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

modelo_rf.fit(X_train, y_train)

y_pred_rf = modelo_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\n================ RESULTADOS RF ================")
print(classification_report(y_test, y_pred_rf))
print(f"Precisión (Accuracy): {acc_rf * 100:.2f}%")

# guardar modelo RF
joblib.dump(modelo_rf, "modelos/modelo_rf_cdc_v1.sav")
print("Modelo RF guardado como modelo_rf_cdc_v1.sav")


print("\n=============================================")
print(" ENTRENAMIENTO COMPLETADO – MODELOS LISTOS")
print("=============================================")
print(f"Precisión RL: {acc_rl * 100:.2f}%")
print(f"Precisión RF: {acc_rf * 100:.2f}%")
