"""
client.py - Cliente de ejemplo para la API REST del modelo Titanic

Demuestra cómo llamar a los endpoints /predict y /train
desde Python de forma programática.

Uso:
    python client.py

Requisito: la API debe estar corriendo antes de ejecutar este script.
    Dentro del contenedor Docker:  python apirest.py
    O con Docker:                  docker run ... titanic-fase3
"""

# requests es la librería para hacer peticiones HTTP desde Python
import requests

# json permite formatear la respuesta para mostrarla más legible
import json

# time permite esperar unos segundos entre llamadas
import time


# ── Configuración ──────────────────────────────────────────────────────────────

# URL base donde está corriendo la API
# Si corres la API dentro de Docker con -p 5000:5000, esta URL es correcta
BASE_URL = "http://localhost:5000"


# ── Función auxiliar para imprimir respuestas ──────────────────────────────────

def imprimir_respuesta(titulo, response):
    """
    Imprime de forma legible el resultado de una llamada a la API.

    Parámetros:
        titulo   : texto descriptivo para identificar la llamada
        response : objeto de respuesta devuelto por requests
    """
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print(f"{'='*60}")
    print(f"  Código HTTP : {response.status_code}")
    # json.dumps con indent=2 formatea el JSON con sangría para que sea legible
    print(f"  Respuesta   :\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}")


# ── 1. Verificar que la API está activa ────────────────────────────────────────

print("\n🔌 Verificando conexión con la API...")

try:
    # GET al endpoint raíz para confirmar que el servidor está vivo
    r = requests.get(BASE_URL, timeout=5)
    imprimir_respuesta("Health Check (GET /)", r)
except requests.exceptions.ConnectionError:
    print(f"\n❌ No se pudo conectar a {BASE_URL}")
    print("   Asegúrate de que la API esté corriendo antes de ejecutar este script.")
    print("   Comando: python apirest.py")
    exit(1)


# ── 2. Llamar a /train ─────────────────────────────────────────────────────────

print("\n\n🏋️  Solicitando entrenamiento del modelo...")

r_train = requests.post(f"{BASE_URL}/train")
imprimir_respuesta("Entrenamiento (POST /train)", r_train)

# Esperamos 15 segundos para que el entrenamiento termine antes de predecir
print("\n⏳ Esperando 15 segundos para que el entrenamiento finalice...")
time.sleep(15)


# ── 3. Llamar a /predict con un solo pasajero ──────────────────────────────────

print("\n\n🔮 Predicción para un solo pasajero...")

# Datos de ejemplo: pasajero de tercera clase, hombre, 22 años
pasajero_1 = {
    "Pclass": 3,
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Sex": "male",
    "Embarked": "S"
}

r_pred1 = requests.post(f"{BASE_URL}/predict", json=pasajero_1)
imprimir_respuesta("Predicción pasajero 1 (POST /predict)", r_pred1)


# ── 4. Llamar a /predict con múltiples pasajeros ───────────────────────────────

print("\n\n🔮 Predicción para múltiples pasajeros...")

# Lista con dos pasajeros distintos para mostrar que acepta lotes
varios_pasajeros = [
    {
        "Pclass": 1,
        "Age": 38.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 71.28,
        "Sex": "female",
        "Embarked": "C"
    },
    {
        "Pclass": 3,
        "Age": 26.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7.92,
        "Sex": "male",
        "Embarked": "S"
    }
]

r_pred2 = requests.post(f"{BASE_URL}/predict", json=varios_pasajeros)
imprimir_respuesta("Predicción múltiples pasajeros (POST /predict)", r_pred2)


# ── 5. Probar un error esperado ────────────────────────────────────────────────

print("\n\n⚠️  Probando manejo de error (datos incompletos)...")

# Enviamos datos sin la columna 'Age' para ver que la API responde con error claro
datos_incompletos = {
    "Pclass": 2,
    "Sex": "female"
    # faltan Age, SibSp, Parch, Fare, Embarked
}

r_error = requests.post(f"{BASE_URL}/predict", json=datos_incompletos)
imprimir_respuesta("Error esperado - datos incompletos (POST /predict)", r_error)

print("\n\n✅ Cliente finalizado correctamente.\n")