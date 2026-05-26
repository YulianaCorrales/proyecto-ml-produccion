# Usa la imagen base de Python 3.11 ligera
FROM python:3.11-slim

# Define el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de dependencias al contenedor
COPY requirements.txt .

# Instala las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el archivo de predicción
COPY predict.py .

# Copia el archivo de entrenamiento
COPY train.py .

# Copia la carpeta de modelos al contenedor
COPY models/ models/

# Crea las carpetas para datos y resultados
RUN mkdir -p data output

# Define el comando por defecto al iniciar el contenedor
CMD ["/bin/bash"]