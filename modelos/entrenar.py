import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

print("=============================================")
print(" CARGANDO DATASET LIMPIO")
print("=============================================")

df = pd.read_csv("datasets/cdc_limpio.csv")

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

#EGRESIÓN LOGÍSTICA

print("\n=============================================")
print(" PREPROCESAMIENTO PARA REGRESIÓN LOGÍSTICA")
print("=============================================")
print("Escalando BMI y Age...")

scaler = StandardScaler()
cols_to_scale = ["BMI", "Age"]

X_train_lr = X_train.copy()
X_test_lr = X_test.copy()

# Entrenar scaler SOLO con TRAIN
scaler.fit(X_train_lr[cols_to_scale])

# Aplicar escalado
X_train_lr[cols_to_scale] = scaler.transform(X_train_lr[cols_to_scale])
X_test_lr[cols_to_scale] = scaler.transform(X_test_lr[cols_to_scale])

# Guardar scaler
joblib.dump(scaler, "modelos/scaler_cdc.sav")


print("\nEntrenando Regresión Logística...")
modelo_rl = LogisticRegression(max_iter=400, class_weight="balanced")
modelo_rl.fit(X_train_lr, y_train)

print("\n============== RESULTADOS REGRESIÓN LOGÍSTICA ==============")
y_pred_rl = modelo_rl.predict(X_test_lr)
#Reporte
print(classification_report(y_test, y_pred_rl))
#PORCENTAJE DE PRECISIÓN
acc_rl = accuracy_score(y_test, y_pred_rl)
print(f"Precisión (Accuracy): {acc_rl * 100:.2f}%")
#Matriz RL
cm_rl = confusion_matrix(y_test, y_pred_rl)
tn, fp, fn, tp = cm_rl.ravel()
labels_rl = [
    [f"TN\n{tn}", f"FP\n{fp}"],
    [f"FN\n{fn}", f"TP\n{tp}"]
]
plt.figure(figsize=(6,5))
sns.heatmap(cm_rl, annot=labels_rl, fmt="", cmap='Blues', cbar=False,
            annot_kws={"size":14})
plt.title("Matriz de Confusión - Regresión Logística")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()
# Guardar modelo
joblib.dump(modelo_rl, "modelos/modelo_rl_cdc.sav")

# RANDOM FOREST
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
print("\n================ RESULTADOS RANDOM FOREST ================")
y_pred_rf = modelo_rf.predict(X_test)
# Reporte
print(classification_report(y_test, y_pred_rf))
#PORCENTAJE DE PRECISIÓN
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Precisión (Accuracy): {acc_rf * 100:.2f}%")
# Matriz RF
cm_rf = confusion_matrix(y_test, y_pred_rf)
tn2, fp2, fn2, tp2 = cm_rf.ravel()
labels_rf = [
    [f"TN\n{tn2}", f"FP\n{fp2}"],
    [f"FN\n{fn2}", f"TP\n{tp2}"]
]
plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=labels_rf, fmt="", cmap='Oranges', cbar=False,
            annot_kws={"size":14})
plt.title("Matriz de Confusión - Random Forest")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()
# Importancia RF
importances = modelo_rf.feature_importances_
df_imp = pd.DataFrame({"Variable": variables, "Importancia": importances}).sort_values("Importancia", ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x="Importancia", y="Variable", data=df_imp, palette="Oranges_r")
plt.title("Importancia de Variables - Random Forest")
plt.show()
# Guardar modelo RF
joblib.dump(modelo_rf, "modelos/modelo_rf_cdc.sav")

print("\n=============================================")
print(" ENTRENAMIENTO COMPLETADO – MODELOS GUARDADOS")
print("=============================================")
print(f"Precisión RL: {acc_rl * 100:.2f}%")
print(f"Precisión RF: {acc_rf * 100:.2f}%")
