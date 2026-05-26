# Extender la imagen de fase-2 que ya tiene Python, pandas, scikit-learn, etc.
# Esto evita repetir toda la configuración base.
FROM titanic-fase2

# Establecer /app como directorio de trabajo (heredado de fase-2, pero lo reafirmamos)
WORKDIR /app

# Copiar el archivo de dependencias nuevas (Flask y requests)
COPY requirements.txt .

# Instalar las nuevas dependencias sobre las que ya tenía la imagen base
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el script de la API REST
COPY apirest.py .

# Copiar el cliente de ejemplo
COPY client.py .

# Declarar que este contenedor usa el puerto 5000
# (el puerto real se mapea con -p 5000:5000 al hacer docker run)
EXPOSE 5000

# Comando por defecto: arrancar la API REST
# Se puede sobreescribir desde docker run si se necesita otra cosa
CMD ["python", "apirest.py"]