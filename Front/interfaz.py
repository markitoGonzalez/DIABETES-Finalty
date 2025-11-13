import streamlit as st
import requests

# CONFIGURACIÓN DE LA APP
st.set_page_config(page_title="Predicción Diabetes", page_icon="🩺")

st.title("🩺 Predicción de Riesgo de Diabetes")

#PRIMERA ETAPA: Nombre del paciente
st.subheader("Datos del Paciente")

nombre = st.text_input("Ingrese el nombre del paciente")

if not nombre:
    st.warning("Por favor ingrese el nombre del paciente para continuar.")
    st.stop()

st.success(f"Paciente seleccionado: {nombre}")

# SEGUNDA ETAPA: Formulario clínico
st.subheader("Formulario Clínico")

opciones = {"Sí": 1, "No": 0}

rango_edades = {
    "18–24": 1, "25–29": 2, "30–34": 3, "35–39": 4,
    "40–44": 5, "45–49": 6, "50–54": 7, "55–59": 8,
    "60–64": 9, "65–69": 10, "70–74": 11, "75–79": 12, "80+": 13
}

salud = {
    "Excelente": 1, "Muy Buena": 2,
    "Buena": 3, "Regular": 4, "Mala": 5
}

col1, col2 = st.columns(2)

with col1:
    HighBP = st.selectbox("Presión Alta", ["Seleccionar..."] + list(opciones.keys()))
    HighChol = st.selectbox("Colesterol Alto", ["Seleccionar..."] + list(opciones.keys()))
    Smoker = st.selectbox("Fumador", ["Seleccionar..."] + list(opciones.keys()))
    Stroke = st.selectbox("Derrame Cerebral", ["Seleccionar..."] + list(opciones.keys()))

with col2:
    HeartDiseaseorAttack = st.selectbox("Enfermedad Cardiaca", ["Seleccionar..."] + list(opciones.keys()))
    PhysActivity = st.selectbox("Actividad Física", ["Seleccionar..."] + list(opciones.keys()))
    DiffWalk = st.selectbox("Dificultad para Caminar", ["Seleccionar..."] + list(opciones.keys()))
    GenHlth = st.selectbox("Salud General", ["Seleccionar..."] + list(salud.keys()))

BMI = st.number_input("Índice de Masa Corporal (BMI)", min_value=10.0, max_value=70.0, value=None, placeholder="Ingrese BMI")

Age = st.selectbox("Rango Etario", ["Seleccionar..."] + list(rango_edades.keys()))

Sexo = st.radio("Sexo", ["Hombre", "Mujer"], index=None)

# BOTÓN DE PREDICCIÓN
if st.button("Predecir"):
    
    # Validación
    if ("Seleccionar..." in [
        HighBP, HighChol, Smoker, Stroke,
        HeartDiseaseorAttack, PhysActivity,
        DiffWalk, GenHlth, Age
    ]) or BMI is None or Sexo is None:

        st.error("⚠️ Debes completar todos los campos antes de predecir.")
        st.stop()

    # Datos a enviar
    datos = {
        "HighBP": opciones[HighBP],
        "HighChol": opciones[HighChol],
        "BMI": BMI,
        "Smoker": opciones[Smoker],
        "Stroke": opciones[Stroke],
        "HeartDiseaseorAttack": opciones[HeartDiseaseorAttack],
        "PhysActivity": opciones[PhysActivity],
        "DiffWalk": opciones[DiffWalk],
        "Sex": 1 if Sexo == "Hombre" else 0,
        "Age": rango_edades[Age],
        "GenHlth": salud[GenHlth]
    }

    # Petición al backend
    url_backend = "http://127.0.0.1:5000/predict"
    r = requests.post(url_backend, json=datos).json()

    if "probabilidad" not in r:
        st.error("Error en el servidor: " + r.get("error", ""))
        st.stop()

    # PROBABILIDAD COMO PORCENTAJE REAL
    prob = r["probabilidad"] * 100
    prob_str = f"{prob:.0f}%"
    st.subheader(f"Resultado para {nombre}")

    # CLASIFICACIÓN SEGÚN PROBABILIDAD
    if prob >= 50:
        st.error(f"**Riesgo Alto de Diabetes**\nProbabilidad: **{prob_str}**")
    elif prob >= 30:
        st.warning(f"**Riesgo Moderado**\nProbabilidad: **{prob_str}**")
    else:
        st.success(f"**Sin Riesgo**\nProbabilidad: **{prob_str}**")
