import pandas as pd
from sklearn.preprocessing import StandardScaler

#CARGAR EL DATASET

print("Cargando dataset CDC")
df = pd.read_csv("datasets/diabetes_012_health_indicators_BRFSS2015.csv")
print("Dataset cargado:", df.shape, "registros cargados.")

#CREAR VARIABLE OBJETIVO BINARIA
print("Creando variable binaria (0=no riesgo, 1=risk)...")
df["Diabetes_binary"] = df["Diabetes_012"].apply(lambda x: 1 if x > 0 else 0)

#ELIMINAR COLUMNAS POCO ÚTILES
columnas_eliminar = [
    "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "Education", "Income", "CholCheck",
    "MentHlth", "PhysHlth"
]

df.drop(columns=columnas_eliminar, inplace=True)
#SELECCIONAR VARIABLES PRINCIPALES

variables_usar = [
    "HighBP", "HighChol", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "DiffWalk",
    "Sex", "Age", "GenHlth"
]
X = df[variables_usar]
y = df["Diabetes_binary"]

print("Variables seleccionadas:", variables_usar)

#LIMPIEZA DE OUTLIERS BÁSICOS
print("Eliminando registros con valores extremos en BMI...")
df = df[(df["BMI"] > 10) & (df["BMI"] < 70)]

#NORMALIZAR VARIABLES CONTINUAS
print("Normalizando BMI y Age...")
scaler = StandardScaler()

df[["BMI", "Age"]] = scaler.fit_transform(df[["BMI", "Age"]])
#GUARDAR DATASET LIMPIO
df_final = df[variables_usar + ["Diabetes_binary"]]
df_final.to_csv("datasets/cdc_limpio.csv", index=False)

print("Limpieza completada.")
print("Archivo guardado como: datasets/cdc_limpio.csv")
print("Registros finales:", df_final.shape)

