# 1. Usar una imagen ligera de Python
FROM python:3.10-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar el archivo de dependencias e instalarlas
# Nota: Asegúrate de tener un archivo requirements.txt (paso siguiente)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar todo el contenido de tu carpeta local al contenedor
# Esto incluye app.py e index.html
COPY . .

# 5. Variable de entorno para el puerto (Cloud Run usa 8080 por defecto)
ENV PORT 8080

# 6. Comando para ejecutar la app usando Gunicorn (servidor de producción)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app