# Verwende ein leichtes Python-Image als Basis
FROM python:3.10-slim

# Setze das Arbeitsverzeichnis im Container
WORKDIR /usr/src/app

# Kopiere die Anwendung in das Arbeitsverzeichnis
COPY . .

# Installiere nur die notwendigen Python-Abhängigkeiten
RUN pip install --no-cache-dir \
    flask \
    transformers \
    boto3 \
    transformers==4.37.0 \
    pillow

# Exponiere den Port, auf dem die Flask-Anwendung läuft
EXPOSE 5000

# Setze den Befehl zum Starten der Anwendung
CMD ["python", "app.py"]