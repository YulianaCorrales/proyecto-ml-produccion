Yuliana Corrales Castaño 

cc: 39193015

Curso: Modelos 1

Universidad de Antioquia

# Proyecto IA listo para producción - Titanic (Kaggle)

Este repositorio contiene el desarrollo de un proyecto de Machine Learning estructurado en tres fases, con el objetivo de llevar un modelo predictivo a un estado listo para ser integrado en un sistema de producción.

El proyecto está basado en la competición de Kaggle:

**Titanic - Machine Learning from Disaster**

---

# Estructura del repositorio

```bash
fase-1/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│
├── modelo.ipynb
├── submission.csv
└── modelo_titanic.pkl


---

# FASE 1 - Modelo Predictivo

## Objetivo
Entrenar un modelo de Machine Learning que prediga si un pasajero sobrevivió o no al Titanic, utilizando el dataset de Kaggle.

Al finalizar esta fase se generan:
- Un archivo `submission.csv` con las predicciones finales.
- Un archivo `modelo_titanic.pkl` con el modelo entrenado (para fases posteriores).

---

## Requisitos
Este proyecto fue desarrollado en **Google Colab**, por lo tanto no requiere instalación manual adicional.

Librerías principales utilizadas:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

---

## Ejecución (Google Colab)

### Paso 1. Descargar los datos desde Kaggle
1. Ingresar a Kaggle y buscar la competición:
   **Titanic - Machine Learning from Disaster**
2. Ir a la pestaña **Data**.
3. Descargar los archivos:
   - `train.csv`
   - `test.csv`
   - `gender_submission.csv`

---

### Paso 2. Subir los datos a Google Colab
1. Abrir el notebook:

   `fase-1/modelo.ipynb`

2. En el panel izquierdo de Colab, seleccionar **Files** (icono de carpeta).
3. Subir manualmente los archivos CSV descargados desde Kaggle.

---

### Paso 3. Ejecutar el notebook
Ejecutar todas las celdas en orden.

Durante la ejecución se realiza:
- Creación de carpetas del proyecto (`data/`, `models/`, `outputs/`)
- Exploración inicial del dataset
- Selección de variables predictoras
- Preprocesamiento (imputación, escalamiento y One-Hot Encoding)
- Entrenamiento del modelo RandomForest
- Evaluación del modelo (accuracy, AUC, matriz de confusión y curva ROC)
- Entrenamiento final con todos los datos
- Generación del archivo `submission.csv`
- Guardado del modelo entrenado en formato `.pkl`

---

## Archivos generados

Al finalizar la ejecución del notebook se generan los siguientes archivos dentro de `fase-1/`:

- `submission.csv`  
  Archivo final con predicciones en el formato requerido por Kaggle.

- `modelo_titanic.pkl`  
  Modelo entrenado guardado para reutilizarlo en la Fase 2 y Fase 3.

---

# FASE 2 - Docker (Pendiente)
En esta fase se construirá un contenedor Docker con los scripts:

- `train.py`: reentrena el modelo y guarda una nueva versión.
- `predict.py`: recibe un archivo CSV y genera predicciones utilizando el modelo guardado.

---

# FASE 3 - API REST (Pendiente)
En esta fase se desarrollará una API REST en Python con endpoints:

- `/predict`: devuelve una predicción para un nuevo dato.
- `/train`: ejecuta un proceso de reentrenamiento del modelo.

---

Proyecto desarrollado como parte del curso de Modelos 1 orientado a producción.
