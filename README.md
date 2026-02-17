# Projet05 - Système de Prédiction RH avec API FastAPI

Un système complet de prédiction machine learning pour l'analyse des données RH (suite du Projet 04
"Cause(s) d'attrition dans une ESN"), avec une API REST déployée via Docker.

## 🎯 Objectif

Ce projet développe une application d'analyse RH permettant de :
- **Prédire** le départ des employés basé sur leurs données
- **Gérer** une base de données des données RH
- **Servir** des prédictions via une API REST
- **Entraîner** et **versionner** des modèles de machine learning

## 📋 Fonctionnalités principales

### 1. **Machine Learning**
- Modèle LogisticRegression pour la classification binaire
- Pipeline de traitement des données avec transformations personnalisées
- Validation croisée StratifiedKFold
- Support des données catégoriques, numériques et binaires

### 2. **API FastAPI**
- Endpoints de prédiction en temps réel
- Gestion des sessions de base de données
- Lifespan asynchrone pour initialisation/fermeture
- Documentation Swagger auto-générée

### 3. **Base de Données**
- Gestion SQL complète hébergé sur supabase
- Tables pour les données d'entrée et les prédictions
- Support PostgreSQL (prod) et SQLite (pytest)
- Scripts sql d'initialisation inclus

### 4. **Tests**
- Suite de tests pytest complète
- Tests API, base de données et modèle
- Couverture de code avec pytest-cov
- Tests asynchrones avec pytest-asyncio

## 🚀 Installation

### Prérequis
- Python >= 3.10
- PostgreSQL (optionnel, SQLite supporté)
- Docker (pour containerisation)

### Installation locale

1. **Cloner le repository**
```bash
git clone <repository-url>
cd projet05_test2
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OU
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configurer la base de données**
```bash
# Modifier les variables d'environnement dans .env si nécessaire
python -c "from src.utils import create_bd_base; create_bd_base()"
```

## 📦 Dépendances principales

| Paquet | Version | Utilisation |
|--------|---------|-----------|
| **FastAPI** | >=0.128.0 | Framework API web |
| **SQLAlchemy** | >=2.0.46 | ORM base de données |
| **scikit-learn** | >=1.8.0 | Machine Learning |
| **pandas** | >=3.0.0 | Manipulation données |
| **numpy** | >=2.4.1 | Calculs numériques |
| **uvicorn** | >=0.30.0 | Serveur ASGI |
| **pytest** | >=9.0.2 | Framework de tests |
| **joblib** | | Sérialisation modèle |

## 📁 Structure du Projet

```
projet05_test2/
├── main.py                # Point d'entrée de l'application FastAPI
├── requirements.txt       # Dépendances du projet
├── pyproject.toml         # Configuration du projet
├── pytest.ini             # Configuration pytest
├── Dockerfile             # Pour containerisation Docker
│
├── src/                   # Code source principal
│   ├── __init__.py
│   ├── bdd.py             # Configuration base de données SQLAlchemy
│   ├── models.py          # Modèles Pydantic pour validation
│   ├── train.py           # Pipeline et entraînement du modèle
│   ├── predict.py         # Chargement et prédiction
│   └── utils.py           # Fonctions utilitaires
│
├── model/                 # Artefacts du modèle
│   ├── ml_model.joblib    # Modèle sérialisé
│   └── __init__.py
│
├── sql/                   # Scripts SQL
│   ├── create_tables_p5_rh.sql
│   ├── extrait_eval_insert.csv
│   ├── extrait_sirh_insert.csv
│   └── extrait_sondage_insert.csv
│
├── test/                  # Suite de tests
│   ├── conftest.py        # Configuration pytest (fixtures)
│   ├── test_api.py        # Tests API
│   ├── test_database.py   # Tests base de données
│   ├── test_model.py      # Tests modèle ML
│   └── test_utils.py      # Tests utilitaires
│
└── htmlcov/               # Rapport de couverture de code
```

## 🔧 Utilisation

### Démarrer l'API en local

```bash
uvicorn main:app --reload --port 8000
```

L'API sera disponible à `http://localhost:8000`
- Documentation Swagger : `http://localhost:8000/docs`
- ReDoc : `http://localhost:8000/redoc`

### Entraîner le modèle
- le script est à lancer en tant que "main"
```python
from src.train import train_model
train_model()
```

### Faire une prédiction

```python
from src.predict import load_model, predict

model = load_model()
prediction = predict(model, data)
```

### Tests

```bash
# Exécuter tous les tests
pytest

# Avec couverture de code
pytest --cov=src --cov-report=html

# Tests spécifiques
pytest test/test_model.py
pytest test/test_api.py -v
```

## 🐳 Docker

### Build et Run avec Docker

```bash
# Construire l'image
docker build -t projet05-rh .

# Lancer le conteneur
docker run -p 7860:7860 projet05-rh
```

L'API sera accessible à `http://localhost:7860`

### Variables d'environnement

Créer un fichier `.env` à la racine pour configurer :

```env
DATABASE_URL=postgresql://user:password@localhost/projet05
PYTHONUNBUFFERED=1
```

## 📊 Modèle de Machine Learning

### Architecture

- **Type** : Classification binaire (LogisticRegression)
- **Target** : Prédiction du départ employé (Oui/Non)

### Features transformées par la pipeline/class :

**Catégoriques (One-Hot Encoded)**:
- `statut_marital` : Marié(e), Célibataire, etc.
- `departement` : Ventes, IT, RH, etc.
- `poste` : Manager, Developer, etc.
- `domaine_etude` : Informatique, Marketing, etc.

**Binaires (0/1)**:
- `genre` : M (1) / F (0)
- `heure_supplementaires` : Oui (1) / Non (0)

**Traitements spéciaux**:
- Transformation pourcentages (% converti en entier)
- Transformation fréquences (Aucun→0, Occasionnel→1, Frequent→2)

## 📝 Endpoints API

### POST `/predict/insert`
Obtenir une prédiction pour un employé

**Request**:
```json
{
  "id_employee": 999,
  "age": 35,
  "genre": "M",
  "revenu_mensuel": 5000,
  "statut_marital": "Marié(e)",
  "departement": "Ventes",
  "poste": "Manager"
}
```

**Response**:
```json
{
  "message": "Input créé et prédiction effectuée avec succès",
  "id_input": 10665,
  "id_employee": 999,
  "prediction": 0
}
```

## 🧪 Tests et Couverture

- **Framework** : pytest + pytest-asyncio
- **Couverture** : Rapports HTML disponibles dans `htmlcov/`
- **Tests unitaires** : Validation des transformations, utilitaires
- **Tests intégration** : API, base de données
- **Tests modèle** : Entraînement et prédiction

Exécuter avec couverture :
```bash
pytest --cov=src --cov-report=html
```

## 🔐 Sécurité

- Validation Pydantic de tous les inputs API
- Dépendances gérées par SQLAlchemy
- Support des variables d'environnement pour config sensible
- Isolation via conteneur Docker

## 📈 Performances

- Pipeline optimisé avec transformations parallélisables
- Sérialisation modèle avec joblib
- API async avec FastAPI/Uvicorn
- Gestion efficace des sessions DB

## 🤝 Contribution

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request


## 📧 Support

Pour toute question ou problème, veuillez ouvrir une issue dans le repository.

---

**Dernière mise à jour** : Février 2026
