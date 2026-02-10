FROM python:3.10-slim

# Évite les fichiers .pyc et force les logs immédiats
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Port utilisé par Hugging Face
EXPOSE 7860

# Lancement FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]