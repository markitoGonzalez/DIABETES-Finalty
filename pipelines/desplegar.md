PIPELINE MLOps COMPLETO — MODELO DIABETES

#1.Preparación del entorno
.\venv\Scripts\activate
pip install -r requirements.txt

#2.Limpieza de datos
python modelos/limpieza_cd.py
# Salida esperada: datasets/processed/cdc_limpio.csv

#3.Entrenamiento del modelo
python modelos/entrenar.py
# Archivos generados:
# modelos/modelo_rl_cdc_v1.sav
# modelos/modelo_rf_cdc_v1.sav
# modelos/scaler_cdc.sav

#4.Pruebas unitarias antes de desplegar
pytest
# Resultado esperado: 3 passed

#5.Despliegue del backend (API Flask)
python app.py
# Endpoints:
# GET /
# POST /predict

#6.Ejecutar interfaz Streamlit
streamlit run interfaz/streamlit_app.py

#7.Reentrenamiento del modelo cuando haya drift o nuevos datos
python modelos/entrenar.py
# Nuevas versiones recomendadas:
# modelo_rl_cdc_v2.sav
# modelo_rl_cdc_v3.sav

#8.Rollback en caso de fallo de una nueva versión
cp modelos/modelo_rl_cdc_v1.sav modelos/modelo_rl_produccion.sav

#9.Checklist final antes del despliegue
# -Limpieza ejecutada 
# -Entrenamiento correcto 
# -Artefactos generados 
# -Pruebas superadas 
# -API funcionando 
# -Streamlit funcionando 
# -Versiones del modelo guardadas 

#10. Diagrama del pipeline (texto)
# Dataset → Limpieza → Entrenamiento → Tests → API Flask → Streamlit → Monitoreo → Reentrenamiento → Rollback
