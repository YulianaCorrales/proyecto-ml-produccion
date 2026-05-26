"""
apirest.py - API REST para el modelo predictivo del Titanic

Expone dos endpoints:
    POST /predict  → recibe datos de un pasajero y devuelve la predicción
    POST /train    → lanza el reentrenamiento del modelo con datos estándar

Uso (dentro del contenedor o local):
    python apirest.py

La API queda disponible en: http://localhost:5000
"""

# Flask es el framework que crea el servidor web
from flask import Flask, request, jsonify

# joblib carga el modelo guardado en disco (.pkl)
import joblib

# pandas maneja los datos en forma de tabla
import pandas as pd

# os permite acceder al sistema de archivos (verificar si existe el modelo)
import os

# threading permite lanzar el entrenamiento en paralelo sin bloquear la API
import threading

# sys y subprocess permiten ejecutar train.py como proceso externo
import subprocess
import sys


# ── Configuración inicial ──────────────────────────────────────────────────────

# Crea la aplicación Flask. __name__ le dice a Flask en qué archivo está.
app = Flask(__name__)

# Ruta donde se guarda el modelo entrenado (la misma que usa fase-2)
MODEL_PATH = "models/titanic_model.pkl"

# Ruta al archivo de datos de entrenamiento estándar
TRAIN_DATA_PATH = "data/train.csv"

# Columnas que el modelo necesita para predecir
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# ── Función auxiliar ───────────────────────────────────────────────────────────

def load_model():
    """
    Carga el modelo desde disco y lo devuelve.
    Si el archivo no existe, devuelve None.
    Esto permite que la API arranque aunque todavía no haya modelo entrenado.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


# ── Endpoint raíz (health check) ───────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    """
    Endpoint de verificación. Sirve para confirmar que la API está viva.
    Al abrir http://localhost:5000 en el navegador verás un mensaje de bienvenida.
    """
    modelo_disponible = os.path.exists(MODEL_PATH)
    return jsonify({
        "mensaje": "API del modelo Titanic funcionando",
        "modelo_disponible": modelo_disponible,
        "endpoints": {
            "POST /predict": "Recibe datos de un pasajero y devuelve predicción",
            "POST /train":   "Lanza reentrenamiento del modelo"
        }
    })


# ── Endpoint /predict ──────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Recibe un JSON con los datos de uno o varios pasajeros y devuelve:
    - survived: 0 (no sobrevivió) o 1 (sobrevivió)
    - probability: probabilidad de sobrevivencia (entre 0 y 1)

    Ejemplo de JSON de entrada (un solo pasajero):
    {
        "Pclass": 3,
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Sex": "male",
        "Embarked": "S"
    }

    También acepta una lista de pasajeros:
    [
        { "Pclass": 1, "Age": 38, ... },
        { "Pclass": 3, "Age": 26, ... }
    ]
    """

    # Cargar el modelo desde disco
    model = load_model()

    # Si no hay modelo todavía, devolver error 503 (servicio no disponible)
    if model is None:
        return jsonify({
            "error": "Modelo no encontrado. Ejecuta primero POST /train para entrenar el modelo."
        }), 503

    # Leer el JSON que envió el cliente
    data = request.get_json()

    # Si no envió nada, devolver error 400 (petición incorrecta)
    if data is None:
        return jsonify({"error": "No se recibieron datos. Envía un JSON válido."}), 400

    # Convertir a DataFrame de pandas.
    # Si es un dict (un solo pasajero), lo envolvemos en lista para que pandas lo acepte.
    if isinstance(data, dict):
        data = [data]

    try:
        df = pd.DataFrame(data)
    except Exception as e:
        return jsonify({"error": f"Error al interpretar los datos: {str(e)}"}), 400

    # Verificar que todas las columnas necesarias estén presentes
    missing = [col for col in ALL_FEATURES if col not in df.columns]
    if missing:
        return jsonify({
            "error": f"Faltan columnas en los datos: {missing}",
            "columnas_requeridas": ALL_FEATURES
        }), 400

    # Seleccionar solo las columnas que el modelo necesita
    X = df[ALL_FEATURES]

    try:
        # Generar predicciones (0 o 1)
        predictions = model.predict(X)

        # Generar probabilidades (número entre 0 y 1)
        probabilities = model.predict_proba(X)[:, 1]
    except Exception as e:
        return jsonify({"error": f"Error durante la predicción: {str(e)}"}), 500

    # Construir la respuesta: una lista de resultados, uno por pasajero
    results = []
    for i in range(len(predictions)):
        results.append({
            "survived": int(predictions[i]),
            "survived_label": "Sobrevivió" if predictions[i] == 1 else "No sobrevivió",
            "probability": round(float(probabilities[i]), 4)
        })

    # Si solo era un pasajero, devolver objeto simple (no lista)
    if len(results) == 1:
        return jsonify(results[0])

    return jsonify(results)


# ── Endpoint /train ────────────────────────────────────────────────────────────

@app.route("/train", methods=["POST"])
def train():
    """
    Lanza el reentrenamiento del modelo usando los datos estándar (data/train.csv).
    El entrenamiento se ejecuta en un hilo separado para no bloquear la API.

    Devuelve inmediatamente un mensaje confirmando que el proceso fue iniciado.
    El modelo quedará actualizado en models/titanic_model.pkl al terminar.
    """

    # Verificar que el archivo de datos de entrenamiento exista
    if not os.path.exists(TRAIN_DATA_PATH):
        return jsonify({
            "error": f"No se encontró el archivo de entrenamiento: {TRAIN_DATA_PATH}"
        }), 404

    def run_training():
        """
        Función interna que ejecuta train.py en un proceso separado.
        Se llama desde un hilo para no bloquear la API mientras entrena.
        """
        print("[/train] Iniciando entrenamiento en segundo plano...")
        # subprocess.run ejecuta un comando de terminal desde Python
        # sys.executable asegura usar el mismo Python del contenedor
        result = subprocess.run(
            [sys.executable, "train.py", TRAIN_DATA_PATH],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("[/train] Entrenamiento completado exitosamente.")
        else:
            print(f"[/train] Error en entrenamiento:\n{result.stderr}")

    # Crear y lanzar el hilo de entrenamiento
    # daemon=True significa que el hilo se cierra si la API se cierra
    hilo = threading.Thread(target=run_training, daemon=True)
    hilo.start()

    return jsonify({
        "mensaje": "Entrenamiento iniciado en segundo plano.",
        "datos_usados": TRAIN_DATA_PATH,
        "modelo_destino": MODEL_PATH,
        "nota": "El modelo estará disponible en unos segundos. Luego puedes usar /predict."
    })


# ── Arranque del servidor ──────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Punto de entrada del script.
    host="0.0.0.0" hace que la API sea accesible desde fuera del contenedor Docker.
    port=5000 es el puerto estándar de Flask.
    debug=False es lo correcto para producción.
    """
    print("=" * 60)
    print("Iniciando API REST - Modelo Titanic")
    print("Disponible en: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)