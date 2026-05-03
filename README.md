
**Nombre**: Yuliana Corrales Castaño  
**Documento de Identidad**: 39193015  
**Institución**: Universidad de Antioquia  
**Proyecto**: ML en Producción 
**Modelo**: Predicción de Supervivencia en el Titanic  

# Proyecto IA listo para producción - Titanic (Kaggle)

Este repositorio contiene el desarrollo de un proyecto de Machine Learning estructurado en tres fases, 
con el objetivo de llevar un modelo predictivo a un estado listo para ser integrado en un sistema de producción.

El proyecto está basado en la competición de Kaggle:

**Titanic - Machine Learning from Disaster**

---
# FASE 1 - Modelo Predictivo

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

## Objetivo
Entrenar un modelo de Machine Learning que prediga si un pasajero sobrevivió o no al Titanic,
utilizando el dataset de Kaggle.

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

# FASE 2 - Docker

markdown# Fase 2: Despliegue en Container Docker

## Descripción General

Esta fase implementa un contenedor Docker completamente funcional que encapsula el modelo predictivo del Titanic junto con sus dependencias. El contenedor permite ejecutar predicciones y reentrenar el modelo de manera aislada y reproducible, garantizando que el sistema funcione de forma idéntica en cualquier máquina o entorno.

## Objetivo

Transformar el modelo entrenado en la Fase 1 en un sistema de producción listo para ser integrado en cualquier plataforma, utilizando Docker como herramienta de containerización. Esto asegura portabilidad, reproducibilidad y facilidad de despliegue.

## Estructura del Directorio
fase-2/
├── Dockerfile                 # Configuración del contenedor Docker
├── requirements.txt          # Dependencias Python necesarias
├── predict.py               # Script para realizar predicciones
├── train.py                 # Script para entrenar/reentrenar el modelo
├── models/
│   └── titanic_model.pkl    # Modelo entrenado (serializado)
├── data/
│   ├── test.csv             # Datos para pruebas/predicción
│   └── train.csv            # Datos para entrenamiento
└── output/
└── predictions.csv      # Resultados de predicciones (generado)

## Tecnologías Utilizadas

- **Python 3.11-slim**: Imagen base ligera para el contenedor
- **Docker**: Containerización
- **Scikit-learn**: Modelo RandomForest y pipeline de preprocesamiento
- **Pandas**: Lectura/escritura de datos
- **NumPy**: Operaciones numéricas
- **Joblib**: Serialización del modelo
- **OneHotEncoder + StandardScaler**: Preprocesamiento de datos

## Dependencias

El archivo `requirements.txt` especifica todas las librerías Python necesarias:

- `pandas==2.1.3`: Manipulación de datos
- `numpy==1.24.3`: Computación numérica
- `scikit-learn==1.3.2`: Algoritmos de machine learning
- `joblib==1.3.2`: Serialización de objetos Python
- `matplotlib==3.8.2`: Visualización (opcional para análisis)
- `seaborn==0.13.0`: Estadísticas visuales (opcional)

## Inicio Rápido

### Requisito Previo
Tener Docker Desktop instalado en tu máquina. Descárgalo desde: https://www.docker.com/products/docker-desktop

### Paso 1: Construir la Imagen Docker

Abre una terminal en la carpeta `fase-2/` y ejecuta:

```bash
docker build -t titanic-model:latest .
```

Esto creará una imagen Docker llamada `titanic-model` con la etiqueta `latest`. El proceso tardará 2-3 minutos la primera vez (descarga la imagen base de Python e instala dependencias).

**Verificar que se creó correctamente:**

```bash
docker images | grep titanic-model
```

Deberías ver algo como:
titanic-model          latest    abc123xyz    2 minutes ago    850MB

### Paso 2: Ejecutar Predicciones

Una vez construida la imagen, puedes hacer predicciones con datos nuevos.

**Comando básico:**

```bash
docker run -v "%cd%\data":/app/data -v "%cd%\output":/app/output --rm titanic-model:latest python predict.py data/test.csv output/predictions.csv
```

**Explicación del comando:**
- `docker run`: Ejecuta un contenedor
- `-v "%cd%\data":/app/data`: Monta la carpeta local `data/` dentro del contenedor
- `-v "%cd%\output":/app/output`: Monta la carpeta local `output/` para guardar resultados
- `--rm`: Elimina el contenedor después de ejecutar (limpia automáticamente)
- `titanic-model:latest`: Imagen a utilizar
- `python predict.py data/test.csv output/predictions.csv`: Comando a ejecutar

**Salida esperada:**
Modelo cargado desde: models/titanic_model.pkl
Datos de entrada cargados: data/test.csv
Dimensiones: (418, 7)
Realizando predicciones...
Predicciones completadas: 418 muestras
Resultados guardados en: output/predictions.csv
RESUMEN DE PREDICCIONES:
Total de registros: 418
Predicciones de sobrevivencia (Survived=1): 140
Predicciones de no sobrevivencia (Survived=0): 278
Probabilidad promedio de sobrevivencia: 0.3456
Predicción completada exitosamente

### Paso 3: Entrenar/Reentrenar el Modelo

Si quieres entrenar un modelo nuevo con datos diferentes:

```bash
docker run -v "%cd%\data":/app/data --rm titanic-model:latest python train.py data/train.csv
```

**Salida esperada:**
======================================================================
ENTRENAMIENTO DEL MODELO TITANIC
Datos cargados: data/train.csv
Dimensiones: (891, 12)
DATOS PREPARADOS:
Features (X): (891, 7)
Target (y): (891,)
Distribución de clases:
- No sobrevivió (0): 549 (61.6%)
- Sobrevivió (1): 342 (38.4%)
DIVISIÓN DE DATOS:
Train: (712, 7)
Validación: (179, 7)
NTRENANDO MODELO...
Modelo entrenado correctamente
EVALUACIÓN EN SET DE VALIDACIÓN:
Accuracy: 0.8100 (81.00%)
AUC ROC: 0.8750
CROSS-VALIDATION (5-Fold):
Scores: ['0.8146', '0.8034', '0.8315', '0.8539', '0.8539']
Promedio: 0.8314 (+/- 0.0210)
GUARDANDO MODELO...
Modelo guardado en: models/titanic_model.pkl
Modelo entrenado exitosamente
Accuracy en validación: 0.8100
AUC ROC: 0.8750
Cross-Validation: 0.8314 (+/- 0.0210)

## Descripción Detallada de Scripts

### `predict.py`

**Propósito**: Lee un CSV con datos del Titanic y genera predicciones de supervivencia.

**Uso**:

```bash
python predict.py <archivo_entrada.csv> <archivo_salida.csv>
```

**Ejemplo**:

```bash
python predict.py data/test.csv output/predictions.csv
```

**Requisitos del CSV de entrada**:

El archivo CSV debe contener las siguientes columnas (en cualquier orden):
- `Pclass` (numérico): Clase del pasajero (1, 2, 3)
- `Age` (numérico): Edad del pasajero
- `SibSp` (numérico): Número de hermanos/cónyuge a bordo
- `Parch` (numérico): Número de padres/hijos a bordo
- `Fare` (numérico): Precio del boleto
- `Sex` (categórico): male / female
- `Embarked` (categórico): S (Southampton), C (Cherbourg), Q (Queenstown)
- `PassengerId` (opcional): ID del pasajero para vincular resultados

**Salida**:

Genera un CSV con tres columnas:
- `PassengerId`: ID del pasajero (si estaba en entrada)
- `Survived`: Predicción binaria (0 = No sobrevivió, 1 = Sobrevivió)
- `Survival_Probability`: Probabilidad de supervivencia (0.0 a 1.0)

**Proceso interno**:

1. Carga el modelo serializado desde `models/titanic_model.pkl`
2. Lee el CSV de entrada
3. Valida que todas las columnas requeridas existan
4. Aplica el pipeline de preprocesamiento (imputación + escalamiento + one-hot encoding)
5. Ejecuta el modelo RandomForest para obtener predicciones
6. Calcula probabilidades de supervivencia
7. Guarda resultados en el CSV de salida

### `train.py`

**Propósito**: Entrena un nuevo modelo RandomForest con datos etiquetados y lo guarda en `models/titanic_model.pkl`.

**Uso**:

```bash
python train.py <archivo_entrenamiento.csv>
```

**Ejemplo**:

```bash
python train.py data/train.csv
```

**Requisitos del CSV de entrenamiento**:

El archivo CSV debe contener:
- `Survived` (numérico): Etiqueta target (0 o 1)
- Todas las columnas requeridas por `predict.py` (Pclass, Age, SibSp, Parch, Fare, Sex, Embarked)

**Proceso de entrenamiento**:

1. Carga datos de entrenamiento
2. Separa features (X) y target (y)
3. Divide datos en 80% entrenamiento y 20% validación (con estratificación)
4. Construye un pipeline con:
   - **Preprocesador numérico**: SimpleImputer (mediana) + StandardScaler
   - **Preprocesador categórico**: SimpleImputer (moda) + OneHotEncoder
   - **Modelo**: RandomForestClassifier (100 árboles, max_depth=15)
5. Entrena el pipeline
6. Evalúa en set de validación (Accuracy, AUC ROC)
7. Realiza cross-validation con 5 folds
8. Guarda el modelo completo en `models/titanic_model.pkl`

**Parámetros del modelo**:

- `n_estimators=100`: Número de árboles en el bosque
- `max_depth=15`: Profundidad máxima de cada árbol
- `min_samples_split=5`: Mínimo de muestras para dividir un nodo
- `random_state=42`: Semilla para reproducibilidad

## Detalles del Dockerfile

```dockerfile
FROM python:3.11-slim
```
Utiliza la imagen oficial de Python 3.11 en su versión slim (menor tamaño, sin herramientas innecesarias).

```dockerfile
WORKDIR /app
```
Establece `/app` como directorio de trabajo dentro del contenedor.

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```
Copia el archivo de dependencias e instala todas las librerías necesarias. El flag `--no-cache-dir` reduce el tamaño de la imagen.

```dockerfile
COPY predict.py .
COPY train.py .
COPY models/ models/
```
Copia los scripts Python y el directorio con el modelo entrenado al contenedor.

```dockerfile
RUN mkdir -p data output
```
Crea los directorios para datos de entrada y salida.

```dockerfile
CMD ["/bin/bash"]
```
Por defecto, abre un shell bash (permite flexibilidad para ejecutar diferentes comandos).

## Preprocesamiento de Datos

El pipeline de preprocesamiento garantiza que los datos se procesen de la misma forma en entrenamiento y predicción:

**Variables numéricas** (Pclass, Age, SibSp, Parch, Fare):
- Imputación: Rellenar valores faltantes con la mediana
- Escalamiento: Normalizar a media=0 y desviación estándar=1

**Variables categóricas** (Sex, Embarked):
- Imputación: Rellenar valores faltantes con la moda (valor más frecuente)
- One-Hot Encoding: Convertir categorías a variables binarias (ej: Sex→Sex_female, Sex_male)

## Formato de Salida

El archivo `predictions.csv` generado tiene el siguiente formato:
PassengerId,Survived,Survival_Probability
892,0,0.11187179487179488
893,0,0.39795634920634915
894,0,0.11485714285714285
...

Donde:
- **PassengerId**: Identificador único del pasajero
- **Survived**: Predicción (0=No, 1=Sí)
- **Survival_Probability**: Confianza de la predicción (0.0 a 1.0)

## 🧪 Pruebas

### Verificar que Docker funciona

```bash
docker --version
```

### Verificar que la imagen se construyó

```bash
docker images
```

### Ejecutar en modo interactivo (para debugging)

```bash
docker run -it -v "%cd%\data":/app/data titanic-model:latest /bin/bash
```

Esto abre un shell dentro del contenedor donde puedes ejecutar comandos manualmente.

### Ver logs

```bash
docker run -v "%cd%\data":/app/data -v "%cd%\output":/app/output titanic-model:latest python predict.py data/test.csv output/predictions.csv 2>&1
```

## 🔐 Consideraciones de Seguridad

- El modelo se incluye en la imagen Docker (2.8 MB), lo cual es seguro pero hace la imagen más grande
- No hay datos sensibles en los CSVs de ejemplo
- El contenedor se ejecuta sin privilegi os de root (heredado de imagen Python)
- Usar `--rm` para eliminar contenedores después de usarlos

## Rendimiento del Modelo

**Métricas en validación**:
- Accuracy: ~81%
- AUC ROC: ~0.87
- Cross-Validation Score: 0.8314 ± 0.0210

El modelo es generalmente bueno, aunque podría mejorarse con feature engineering o tuning de hiperparámetros.

## Troubleshooting

### Error: "No such file or directory"

Asegúrate de que:
- Estés en la carpeta `fase-2/`
- Los archivos `data/test.csv` y `data/train.csv` existan
- La carpeta `output/` existe (se crea automáticamente)

### Error: "Dockerfile cannot be empty"

El archivo Dockerfile existe pero está vacío. Verifica que tenga contenido.

### Error: "ModuleNotFoundError: No module named 'pandas'"

Reconstruye la imagen sin cache:

```bash
docker build --no-cache -t titanic-model:latest .
```

### Las predicciones son todas iguales

Verifica que el CSV de entrada tenga las 7 columnas requeridas con los nombres exactos.

## Referencias

- **Docker Documentation**: https://docs.docker.com/
- **Scikit-learn Pipeline**: https://scikit-learn.org/stable/modules/compose.html
- **Titanic Dataset**: https://www.kaggle.com/c/titanic

---

**Estado del Proyecto**: ✅ Completado y Funcional


# FASE 3 - API REST (Pendiente)
En esta fase se desarrollará una API REST en Python con endpoints:

- `/predict`: devuelve una predicción para un nuevo dato.
- `/train`: ejecuta un proceso de reentrenamiento del modelo.

---

Proyecto desarrollado como parte del curso de Modelos 1 orientado a producción.
