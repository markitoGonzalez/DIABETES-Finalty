import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

print("=== INICIO DEL PROCESO DE LIMPIEZA ===")

# ============================
# 1. CARGAR DATASET RAW
# ============================
ruta_raw = "datasets/raw/diabetes_012_health_indicators_BRFSS2015.csv"

if not os.path.exists(ruta_raw):
    raise FileNotFoundError(f"ERROR: No se encontró el archivo en {ruta_raw}")

print("Cargando dataset desde:", ruta_raw)
df = pd.read_csv(ruta_raw)
print(f"Dataset cargado correctamente: {df.shape[0]} registros, {df.shape[1]} columnas.")


# ============================
# 2. CREAR VARIABLE OBJETIVO
# ============================
print("Creando variable objetivo binaria Diabetes_binary...")
df["Diabetes_binary"] = df["Diabetes_012"].apply(lambda x: 1 if x > 0 else 0)


# ============================
# 3. ELIMINAR COLUMNAS POCO ÚTILES
# ============================
columnas_eliminar = [
    "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "Education", "Income", "CholCheck",
    "MentHlth", "PhysHlth"
]

df.drop(columns=columnas_eliminar, inplace=True, errors="ignore")
print("Columnas eliminadas:", columnas_eliminar)


# ============================
# 4. SELECCIONAR VARIABLES PRINCIPALES
# ============================
variables_usar = [
    "HighBP", "HighChol", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "DiffWalk",
    "Sex", "Age", "GenHlth"
]

print("Variables seleccionadas:", variables_usar)


# ============================
# 5. ELIMINAR OUTLIERS SIMPLES
# ============================
print("Filtrando valores extremos en BMI...")
df = df[(df["BMI"] > 10) & (df["BMI"] < 70)]
print("Registros restantes:", df.shape[0])


# ============================
# 6. NORMALIZACIÓN (BMI y Age)
# ============================
print("Normalizando variables continuas (BMI, Age)...")
scaler = StandardScaler()
df[["BMI", "Age"]] = scaler.fit_transform(df[["BMI", "Age"]])


# ============================
# 7. CREAR DATASET FINAL
# ============================
df_final = df[variables_usar + ["Diabetes_binary"]]


# ============================
# 8. GUARDAR DATASET LIMPIO EN processed/
# ============================
ruta_salida = "datasets/processed/cdc_limpio.csv"
df_final.to_csv(ruta_salida, index=False)

print("Limpieza completada correctamente.")
print("Archivo guardado en:", ruta_salida)
print("Registros finales:", df_final.shape)
print("=== FIN DEL PROCESO DE LIMPIEZA ===")
