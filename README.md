
**Nombre**: Yuliana Corrales Castaño  
**Documento de Identidad**: 39193015  
**Institución**: Universidad de Antioquia  
**Proyecto**: ML en Producción 
**Modelo**: Predicción de Supervivencia en el Titanic  

---

# Proyecto IA listo para producción — Titanic (Kaggle)

Este repositorio contiene el desarrollo de un proyecto de Machine Learning estructurado en tres fases, con el objetivo de llevar un modelo predictivo a un estado listo para ser integrado en un sistema de producción.

El proyecto está basado en la competición de Kaggle:
**Titanic - Machine Learning from Disaster**

---

# Estructura del repositorio

```
├── fase-1/
│   ├── data/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── gender_submission.csv
│   ├── modelo.ipynb
│   ├── submission.csv
│   ├── titanic_model.pkl
│   └── README.md
│
├── fase-2/
│   ├── data/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── models/
│   │   └── titanic_model.pkl
│   ├── output/
│   │   └── predictions.csv
│   ├── train.py
│   ├── predict.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── fase-3/
│   ├── data/
│   │   └── train.csv
│   ├── models/
│   │   └── titanic_model.pkl
│   ├── output/
│   ├── train.py
│   ├── predict.py
│   ├── apirest.py
│   ├── client.py
│   ├── Dockerfile
│   └── requirements.txt
│
└── README.md
```

---

# FASE 1 — Modelo Predictivo

## Objetivo

Entrenar un modelo de Machine Learning que prediga si un pasajero sobrevivió o no al hundimiento del Titanic, usando el dataset de la competición de Kaggle.

## Archivos

| Archivo | Descripción |
|---|---|
| `modelo.ipynb` | Notebook con todo el flujo: exploración, preprocesamiento, entrenamiento y evaluación |
| `data/train.csv` | Datos de entrenamiento con etiquetas |
| `data/test.csv` | Datos de prueba sin etiquetas |
| `data/gender_submission.csv` | Ejemplo de formato de entrega requerido por Kaggle |
| `submission.csv` | Predicciones finales generadas para Kaggle |
| `titanic_model.pkl` | Modelo entrenado guardado en disco |

## Requisitos

El notebook fue desarrollado en **Google Colab**, que ya incluye todas las librerías necesarias:

- `pandas`, `numpy` — manejo y transformación de datos
- `scikit-learn` — modelo predictivo y preprocesamiento
- `matplotlib`, `seaborn` — visualización de resultados
- `joblib` — guardado y carga del modelo

## Pasos para ejecutar

### Paso 1 — Descargar los datos desde Kaggle

1. Ir a [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)
2. Abrir la pestaña **Data**
3. Descargar los tres archivos: `train.csv`, `test.csv`, `gender_submission.csv`

### Paso 2 — Abrir el notebook en Google Colab

1. Ir a [https://colab.research.google.com](https://colab.research.google.com)
2. Cargar el archivo `fase-1/modelo.ipynb` desde tu computadora

### Paso 3 — Subir los datos al entorno de Colab

1. En el panel izquierdo de Colab, hacer clic en el ícono de carpeta
2. Subir los tres archivos CSV descargados en el Paso 1

### Paso 4 — Ejecutar el notebook

Ir a **Entorno de ejecución > Ejecutar todo** para correr todas las celdas en orden.

El notebook realiza automáticamente:

- Exploración inicial del dataset (dimensiones, valores faltantes, estadísticas)
- Selección de variables predictoras numéricas y categóricas
- Preprocesamiento: imputación de valores faltantes, escalamiento y One-Hot Encoding
- Entrenamiento de un modelo `RandomForestClassifier`
- Evaluación con accuracy, AUC-ROC, matriz de confusión y curva ROC
- Reentrenamiento final usando el 100% de los datos
- Generación del archivo `submission.csv`
- Guardado del modelo en `titanic_model.pkl`

## Archivos generados

- `submission.csv` — predicciones en el formato requerido por Kaggle
- `titanic_model.pkl` — modelo entrenado, reutilizado en Fase 2 y Fase 3

---

# FASE 2 — Despliegue en contenedor Docker

## Objetivo

Convertir el modelo en un flujo reproducible y portable usando Docker. Los scripts `train.py` y `predict.py` permiten entrenar el modelo y generar predicciones desde dentro de un contenedor, sin depender del entorno local.

## Archivos

| Archivo | Descripción |
|---|---|
| `train.py` | Entrena el modelo con un CSV y guarda el resultado en `models/titanic_model.pkl` |
| `predict.py` | Lee un CSV de entrada y escribe las predicciones en otro CSV |
| `Dockerfile` | Define la imagen Docker con Python y todas las dependencias |
| `requirements.txt` | Lista de librerías Python necesarias |
| `data/train.csv` | Datos de entrenamiento |
| `data/test.csv` | Datos de prueba para predecir |
| `models/titanic_model.pkl` | Modelo preentrenado incluido en el repositorio |
| `output/predictions.csv` | Predicciones generadas por `predict.py` |

## Requisitos

Tener instalado **Docker Desktop**: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

## Pasos para ejecutar

### Paso 1 — Ir a la carpeta fase-2

```bash
cd fase-2
```

### Paso 2 — Construir la imagen Docker

```bash
docker build -t titanic-fase2 .
```

Este comando lee el `Dockerfile`, instala Python 3.11 y todas las librerías del `requirements.txt`, y copia los scripts al contenedor. La primera vez puede tardar unos minutos.

### Paso 3 — Entrenar el modelo dentro del contenedor

En **Linux / Mac:**
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  titanic-fase2 python train.py data/train.csv
```

En **Windows (CMD):**
```bash
docker run --rm -v "%cd%/data:/app/data" -v "%cd%/models:/app/models" titanic-fase2 python train.py data/train.csv
```

El modelo entrenado se guarda en `fase-2/models/titanic_model.pkl`.

### Paso 4 — Generar predicciones dentro del contenedor

En **Linux / Mac:**
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/output:/app/output" \
  titanic-fase2 python predict.py data/test.csv output/predictions.csv
```

En **Windows (CMD):**
```bash
docker run --rm -v "%cd%/data:/app/data" -v "%cd%/models:/app/models" -v "%cd%/output:/app/output" titanic-fase2 python predict.py data/test.csv output/predictions.csv
```

Las predicciones quedan guardadas en `fase-2/output/predictions.csv`.

### (Opcional) Ejecución local sin Docker

```bash
pip install -r requirements.txt
python train.py data/train.csv
python predict.py data/test.csv output/predictions.csv
```

---

# FASE 3 — API REST

## Objetivo

Exponer el modelo como una API REST usando Flask, de modo que cualquier aplicación externa pueda hacer predicciones o lanzar un reentrenamiento enviando una petición HTTP.

## Archivos

| Archivo | Descripción |
|---|---|
| `apirest.py` | Servidor Flask con los endpoints `/predict` y `/train` |
| `client.py` | Script Python que llama a la API y muestra los resultados |
| `train.py` | Script de fase-2 reutilizado por la API para reentrenar el modelo |
| `predict.py` | Script de fase-2 reutilizado por la API para generar predicciones |
| `Dockerfile` | Extiende la imagen de fase-2 agregando Flask y requests |
| `requirements.txt` | Dependencias nuevas respecto a fase-2: Flask y requests |

## Endpoints disponibles

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Verifica que la API está activa y si hay modelo disponible |
| `POST` | `/predict` | Recibe datos de un pasajero en JSON y devuelve la predicción |
| `POST` | `/train` | Lanza el reentrenamiento del modelo en segundo plano |

## Requisitos

- Haber construido la imagen `titanic-fase2` del paso anterior (fase-3 la extiende)
- Docker Desktop instalado

## Pasos para ejecutar

### Paso 1 — Construir primero la imagen base de fase-2

> Si ya la construiste anteriormente, puedes saltar este paso.

```bash
cd fase-2
docker build -t titanic-fase2 .
cd ..
```

### Paso 2 — Ir a la carpeta fase-3

```bash
cd fase-3
```

### Paso 3 — Construir la imagen de fase-3

```bash
docker build -t titanic-fase3 .
```

### Paso 4 — Arrancar la API dentro del contenedor

En **Linux / Mac:**
```bash
docker run --rm -it \
  -p 5000:5000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  titanic-fase3
```

En **Windows (CMD):**
```bash
docker run --rm -it -p 5000:5000 -v "%cd%/data:/app/data" -v "%cd%/models:/app/models" titanic-fase3
```

La API queda disponible en: `http://localhost:5000`

Deberías ver en la terminal:
```
============================================================
Iniciando API REST - Modelo Titanic
Disponible en: http://localhost:5000
============================================================
```

Puedes verificar que funciona abriendo `http://localhost:5000` en el navegador. La respuesta esperada es:

```json
{
  "endpoints": {
    "POST /predict": "Recibe datos de un pasajero y devuelve predicción",
    "POST /train": "Lanza reentrenamiento del modelo"
  },
  "mensaje": "API del modelo Titanic funcionando",
  "modelo_disponible": false
}
```

> `modelo_disponible: false` es normal en este punto. El modelo se activa después de llamar a `/train`.

### Paso 5 — Ejecutar el cliente en otra terminal

Con la API corriendo, abrir una **segunda terminal** y ejecutar:

```bash
cd fase-3
pip install requests
python client.py
```

El script `client.py` realiza automáticamente estos pasos:

1. Verifica que la API esté activa (`GET /`)
2. Solicita el reentrenamiento del modelo (`POST /train`)
3. Espera 15 segundos a que el entrenamiento finalice
4. Hace una predicción con un pasajero individual (`POST /predict`)
5. Hace una predicción con varios pasajeros a la vez (`POST /predict`)
6. Prueba el manejo de errores enviando datos incompletos

### Paso 6 — Probar los endpoints manualmente con curl (opcional)

**Health check:**
```bash
curl http://localhost:5000/
```

**Lanzar entrenamiento:**
```bash
curl -X POST http://localhost:5000/train
```

**Predicción individual:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass": 3, "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Sex": "male", "Embarked": "S"}'
```

## Ejemplo de respuesta de `/predict`

```json
{
  "survived": 0,
  "survived_label": "No sobrevivió",
  "probability": 0.1423
}
```

---

Proyecto desarrollado como parte del curso Modelos 1 — Universidad de Antioquia.
