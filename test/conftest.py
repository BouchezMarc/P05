import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

# Ajouter le répertoire parent au path pour importer les modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Base, Pred, TInputs, ViewInputs
from main import app
from fastapi.testclient import TestClient


# Créer une BD de test en mémoire
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function", autouse=True)
def mock_model():
    """Mock le modèle ML pour les tests"""
    mock = MagicMock()
    mock.predict.return_value = np.array([1])  # Retourne toujours "Oui"
    
    with patch("main.model", mock):
        yield mock


@pytest.fixture(scope="function")
def db_session():
    """Fixture pour obtenir une session de BD test"""
    # Créer les tables avant chaque test
    Base.metadata.create_all(bind=engine)
    
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()
    
    # Nettoyer les tables
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Fixture pour créer un client de test FastAPI"""
    from src.bdd import get_db
    
    # Override la dépendance get_db avec un générateur
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Nettoyer après le test
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def sample_input_data(db_session):
    """Fixture pour créer un input d'exemple en BD"""
    # Créer dans TInputs
    input_record = TInputs(
        id_employee=100,
        age=35,
        genre="M",
        revenu_mensuel=5000,
        statut_marital="Marié(e)",
        departement="Ventes",
        poste="Manager",
        nombre_experiences_precedentes=3,
        annees_dans_l_entreprise=5,
        annees_dans_le_poste_actuel=2,
        satisfaction_employee_environnement=3,
        note_evaluation_precedente=4,
        satisfaction_employee_nature_travail=4,
        satisfaction_employee_equipe=4,
        satisfaction_employee_equilibre_pro_perso=3,
        heure_supplementaires="Oui",
        augementation_salaire_precedente="15 %",
        a_quitte_l_entreprise="Non",
        nombre_participation_pee=2,
        nb_formations_suivies=3,
        distance_domicile_travail=10,
        niveau_education=3,
        domaine_etude="Commerce",
        frequence_deplacement="Frequent",
        annees_depuis_la_derniere_promotion=1,
        annee_experience_totale=10,
        niveau_hierarchique_poste=3
    )
    db_session.add(input_record)
    db_session.commit()
    db_session.refresh(input_record)
    
    # Créer aussi dans ViewInputs pour les tests (simuler la vue)
    view_record = ViewInputs(
        id_input=input_record.id_input,
        id_employee=100,
        age=35,
        genre="M",
        revenu_mensuel=5000,
        statut_marital="Marié(e)",
        departement="Ventes",
        poste="Manager",
        nombre_experiences_precedentes=3,
        annees_dans_l_entreprise=5,
        annees_dans_le_poste_actuel=2,
        satisfaction_employee_environnement=3,
        note_evaluation_precedente=4,
        satisfaction_employee_nature_travail=4,
        satisfaction_employee_equipe=4,
        satisfaction_employee_equilibre_pro_perso=3,
        heure_supplementaires=1,  # Transformé en int
        augementation_salaire_precedente=15,  # Transformé en int
        a_quitte_l_entreprise="Non",
        nombre_participation_pee=2,
        nb_formations_suivies=3,
        distance_domicile_travail=10,
        niveau_education=3,
        domaine_etude="Commerce",
        frequence_deplacement="Frequent",
        annees_depuis_la_derniere_promotion=1,
        annee_experience_totale=10,
        niveau_hierarchique_poste=3,
        satisfaction_globale=3.5,
        dispersion_satisfaction=0.5,
        ratio_fidelite=0.4,
        ratio_stagnation_poste=0.5,
        duree_moyenne_experience=2.0,
        salaire_par_annee_experience=500.0,
        salaire_vs_poste=1.0,
        salaire_vs_niveau=1.0
    )
    db_session.add(view_record)
    db_session.commit()
    
    return input_record
