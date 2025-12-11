#PIPELINE CI/CD — MODELO DE DIABETES (MLOps)

#OBJETIVO
Automatizar el proceso de pruebas, entrenamiento, versionado y despliegue del modelo utilizando un flujo CI/CD simplificado.

# 1.Evento que inicia la ejecución (trigger)
# Cuando se hace un commit o push en el repositorio:
# - Cambios en modelos/
# - Cambios en app.py
# - Cambios en scripts de limpieza o entrenamiento

# 2.Instalar dependencias
.\venv\Scripts\activate
pip install -r requirements.txt

# 3.Ejecutar pruebas unitarias (CI)
pytest
# Si falla - detener pipeline
# Si pasa - continuar

# 4.Ejecutar limpieza de datos (solo si hay cambios en el dataset)
python modelos/limpieza_cd.py

# 5.Entrenamiento del modelo (CI)
python modelos/entrenar.py
# Artefactos generados:
# modelos/modelo_rl_cdc_vX.sav
# modelos/scaler_cdc.sav

# 6.Versionado automático del modelo
# Reglas:
# - Crear nueva versión incremental:
#   modelo_rl_cdc_v1.sav → modelo_rl_cdc_v2.sav → modelo_rl_cdc_v3.sav
# - Guardar historial en /modelos/

# 7.Validación pos-entrenamiento (CI)
pytest
# Se revisa:
# - Predicciones válidas
# - Probabilidad válida
# - Integridad del artefacto

# 8.Despliegue controlado del backend (CD)
# Reiniciar servidor con nuevo modelo:
python app.py

# 9. Validación de la API después del despliegue
# Prueba automática:
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "
{
  \"HighBP\": 1,
  \"HighChol\": 0,
  \"BMI\": 30,
  \"Smoker\": 1,
  \"Stroke\": 0,
  \"HeartDiseaseorAttack\": 0,
  \"PhysActivity\": 1,
  \"DiffWalk\": 0,
  \"Sex\": 1,
  \"Age\": 6,
  \"GenHlth\": 3
}"
# Si responde correctamente → despliegue aprobado

# 10. Publicación del modelo en producción
# Modelo final activo:
cp modelos/modelo_rl_cdc_vX.sav modelos/modelo_rl_produccion.sav

# 11. Rollback automático en caso de falla
# Si el endpoint falla, restaurar versión anterior:
cp modelos/modelo_rl_cdc_v(X-1).sav modelos/modelo_rl_produccion.sav

# 12. Notificación al equipo (simulada)
# "El despliegue ha sido completado correctamente."
# o
# "Rollback activado por error en producción."

# DIAGRAMA DEL FLUJO CI/CD
# Commit → Instalar dependencias → Pruebas → Limpieza → Entrenamiento → Versionado → Pruebas → Deploy → Validación → Aprobado/rollback

