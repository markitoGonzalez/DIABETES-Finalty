import streamlit as st
import requests

st.set_page_config(page_title="Predicción Diabetes", page_icon="🩺")

st.title("🩺 Predicción de Riesgo de Diabetes")

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
    HighBP = opciones[st.selectbox("Presión Alta", opciones)]
    HighChol = opciones[st.selectbox("Colesterol Alto", opciones)]
    Smoker = opciones[st.selectbox("Fumador", opciones)]
    Stroke = opciones[st.selectbox("Derrame Cerebral", opciones)]

with col2:
    HeartDiseaseorAttack = opciones[st.selectbox("Enfermedad Cardiaca", opciones)]
    PhysActivity = opciones[st.selectbox("Actividad Física", opciones)]
    DiffWalk = opciones[st.selectbox("Dificultad para Caminar", opciones)]
    GenHlth = salud[st.selectbox("Salud General", salud)]

BMI = st.number_input("Índice de Masa Corporal (BMI)", 10.0, 70.0, 25.0)
Age = rango_edades[st.selectbox("Rango Etario", rango_edades.keys())]

Sexo = st.radio("Sexo", ["Hombre", "Mujer"])
Sex = 1 if Sexo == "Hombre" else 0

if st.button("Predecir"):

    datos = {
        "HighBP": HighBP,
        "HighChol": HighChol,
        "BMI": BMI,
        "Smoker": Smoker,
        "Stroke": Stroke,
        "HeartDiseaseorAttack": HeartDiseaseorAttack,
        "PhysActivity": PhysActivity,
        "DiffWalk": DiffWalk,
        "Sex": Sex,
        "Age": Age,
        "GenHlth": GenHlth
    }

    r = requests.post("http://127.0.0.1:5000/predict", json=datos).json()

    if "probabilidad" not in r:
        st.error("Error en servidor: " + r.get("error", ""))
    else:
        if r["resultado"] == 1:
            st.error(f"Riesgo de diabetes — prob: {r['probabilidad']:.2f}")
        else:
            st.success(f"Sin riesgo — prob: {r['probabilidad']:.2f}")
