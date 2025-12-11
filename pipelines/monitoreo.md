# MONITOREO DEL MODELO EN PRODUCCIÓN — PIPELINE MLOps

# 1. Métricas Técnicas
# - Latencia del endpoint /predict (objetivo: < 200ms)
# - Cantidad de requests por minuto/hora
# - Errores HTTP 4xx y 5xx
# - Fallos del modelo o scaler

# 2. Métricas del Modelo
# - Distribución de probabilidades (detectar cambios bruscos)
# - Comportamiento anómalo (modelo devuelve siempre 0 o siempre 1)
# - Valores extremos en variables clínicas

# 3. Data Drift
# Variables monitoreadas:
# - BMI
# - Age
# - HighBP
# - GenHlth
#
# Señales de drift:
# - Cambio >15% en promedio de BMI
# - Cambio >10% en promedio de Age
# - Cambios drásticos en proporciones de categorías 0/1
#
# Ejemplo de alerta:
# "ALERTA: BMI promedio cambió de 28.5 a 34.1 — Drift detectado."

# 4. Logs desde la API (app.py)
# Se registran:
# - Requests recibidos
# - Inputs utilizados en predicción
# - Latencia del modelo
# - Errores del sistema o del modelo
#
# Ejemplos:
# [INFO] Predicción realizada — Latencia: 0.03s
# [WARNING] Input fuera de rango: BMI=95
# [ERROR] scaler.transform falló — valores inesperados

# 5. Alertas que activan acciones
# - Aumento sostenido de latencia
# - Probabilidades anormales (siempre iguales)
# - Inputs inválidos repetidos
# - Caída del endpoint
# - Drift confirmado en BMI o Age
# - Errores continuos de predicción

# 6. Acciones Correctivas
# Reentrenamiento:
python modelos/entrenar.py

# Rollback inmediato:
# Restaurar modelo anterior estable:
cp modelos/modelo_rl_cdc_v(X-1).sav modelos/modelo_rl_produccion.sav

# Ajustes recomendados:
# - Validar rangos de entrada
# - Registrar más logs
# - Analizar patrones de drift

# 7. Ciclo Completo de Monitoreo
# 1. Registrar métricas → 2. Analizar latencia y drift → 3. Detectar fallas →
# 4. Decidir reentrenamiento o rollback → 5. Registrar nueva versión del modelo.

# 8. Diagrama de Monitoreo (texto)
# API → Logs → Análisis → Detección de Drift/Errores → Reentrenamiento/rollback
