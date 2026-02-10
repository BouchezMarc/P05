import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import ModelHandler, transform_binary, transform_percent, transform_freq
from src.predict import load_model


class TestTransformations:
    """Tests des fonctions de transformation"""
    
    def test_transform_binary(self):
        """Test transformation binaire Oui/Non"""
        data = pd.DataFrame(
            {
                "genre": ["F", "M", "F"],
                "heure_supplementaires": ["Non", "Oui", "Oui"],
            }
        )
        result = transform_binary(data)
        assert result["genre"].tolist() == [0, 1, 0]
        assert result["heure_supplementaires"].tolist() == [0, 1, 1]
    
    def test_transform_percent(self):
        """Test transformation pourcentage"""
        data = pd.DataFrame({"augementation_salaire_precedente": ["10 %", "25 %", "0 %"]})
        result = transform_percent(data)
        assert result.iloc[:, 0].tolist() == [10, 25, 0]
    
    def test_transform_freq(self):
        """Test transformation fréquence"""
        data = pd.DataFrame({"frequence_deplacement": ["Aucun", "Occasionnel", "Frequent"]})
        result = transform_freq(data)
        assert result["frequence_deplacement"].tolist() == [0, 1, 2]


class TestModelHandler:
    """Tests du ModelHandler"""
    
    def test_build_preprocessor(self):
        """Test création du préprocesseur"""
        handler = ModelHandler(model=LogisticRegression(max_iter=1000))
        preprocessor = handler._build_preprocessor()
        assert preprocessor is not None
    
    def test_build_pipeline(self):
        """Test création de la pipeline"""
        handler = ModelHandler(model=LogisticRegression(max_iter=1000))
        pipeline = handler.build_pipeline()
        assert pipeline is not None
    
    def test_predict_with_dummy_data(self):
        """Test prédiction avec données factices"""
        handler = ModelHandler(model=LogisticRegression(max_iter=1000))
        pipeline = handler.build_pipeline()
        
        # Données factices compatibles avec les colonnes attendues
        dummy_data = pd.DataFrame([
            {
                "age": 35,
                "genre": "M",
                "revenu_mensuel": 5000,
                "statut_marital": "Marié(e)",
                "departement": "Ventes",
                "poste": "Manager",
                "nombre_experiences_precedentes": 3,
                "annees_dans_l_entreprise": 5,
                "note_evaluation_precedente": 4,
                "heure_supplementaires": "Oui",
                "augementation_salaire_precedente": "15 %",
                "nombre_participation_pee": 2,
                "nb_formations_suivies": 3,
                "distance_domicile_travail": 10,
                "niveau_education": 3,
                "domaine_etude": "Commerce",
                "frequence_deplacement": "Frequent",
                "annees_depuis_la_derniere_promotion": 1,
                "niveau_hierarchique_poste": 3,
                "annee_experience_totale": 10,
                "satisfaction_globale": 3.5,
                "dispersion_satisfaction": 0.5,
                "ratio_fidelite": 0.4,
                "ratio_stagnation_poste": 0.5,
                "duree_moyenne_experience": 2.0,
                "salaire_par_annee_experience": 500.0,
                "salaire_vs_poste": 1.0,
                "salaire_vs_niveau": 1.0
            }
        ])
        
        # Entraîner pipeline avec 2 classes
        X_train = pd.concat([dummy_data, dummy_data], ignore_index=True)
        y_train = np.array([0, 1])
        pipeline.fit(X_train, y_train)
        
        # Prédire
        prediction = pipeline.predict(dummy_data)
        assert prediction is not None
        assert prediction[0] in [0, 1]
    
    def test_train_model(self):
        """Test entraînement du modèle avec StratifiedKFold"""
        handler = ModelHandler(model=LogisticRegression(max_iter=1000), n_splits=2)
        pipeline = handler.build_pipeline()
        
        # Créer des données d'entraînement
        X_train = pd.DataFrame({
            "age": [30, 35, 40, 45, 50, 55],
            "genre": ["M", "F", "M", "F", "M", "F"],
            "revenu_mensuel": [4000, 5000, 6000, 4500, 5500, 6500],
            "statut_marital": ["Marié(e)", "Célibataire", "Marié(e)", "Célibataire", "Marié(e)", "Veuf(ve)"],
            "departement": ["IT", "RH", "Ventes", "IT", "RH", "Ventes"],
            "poste": ["Dev", "Manager", "Manager", "Dev", "Manager", "Dev"],
            "nombre_experiences_precedentes": [2, 3, 5, 2, 4, 3],
            "annees_dans_l_entreprise": [3, 5, 7, 2, 6, 4],
            "note_evaluation_precedente": [4, 5, 3, 4, 5, 3],
            "heure_supplementaires": ["Oui", "Non", "Oui", "Non", "Oui", "Non"],
            "augementation_salaire_precedente": ["15 %", "20 %", "10 %", "15 %", "20 %", "10 %"],
            "nombre_participation_pee": [2, 3, 1, 2, 3, 1],
            "nb_formations_suivies": [3, 4, 2, 3, 4, 2],
            "distance_domicile_travail": [10, 15, 5, 10, 15, 5],
            "niveau_education": [3, 4, 5, 3, 4, 5],
            "domaine_etude": ["Commerce", "Gestion", "IT", "Commerce", "Gestion", "IT"],
            "frequence_deplacement": ["Frequent", "Occasionnel", "Aucun", "Frequent", "Occasionnel", "Aucun"],
            "annees_depuis_la_derniere_promotion": [1, 2, 0, 1, 2, 0],
            "niveau_hierarchique_poste": [2, 3, 4, 2, 3, 4],
            "annee_experience_totale": [8, 10, 15, 7, 12, 10],
            "satisfaction_globale": [3.5, 4.0, 3.0, 3.5, 4.0, 3.0],
            "dispersion_satisfaction": [0.5, 0.3, 0.7, 0.5, 0.3, 0.7],
            "ratio_fidelite": [0.4, 0.6, 0.3, 0.4, 0.6, 0.3],
            "ratio_stagnation_poste": [0.5, 0.2, 0.8, 0.5, 0.2, 0.8],
            "duree_moyenne_experience": [2.0, 2.5, 3.0, 2.0, 2.5, 3.0],
            "salaire_par_annee_experience": [500.0, 500.0, 400.0, 500.0, 500.0, 400.0],
            "salaire_vs_poste": [1.0, 1.1, 0.9, 1.0, 1.1, 0.9],
            "salaire_vs_niveau": [1.0, 1.0, 1.1, 1.0, 1.0, 1.1]
        })
        y_train = pd.Series([0, 1, 0, 1, 0, 1])
        
        # Entraîner
        handler.train_model(X_train, y_train)
        
        # Vérifier que la pipeline a été entraînée
        assert hasattr(handler, 'pipeline')
        assert handler.pipeline is not None
    
    def test_evaluate_model(self):
        """Test évaluation du modèle"""
        handler = ModelHandler(model=LogisticRegression(max_iter=1000))
        pipeline = handler.build_pipeline()
        
        # Données de train
        X_train = pd.DataFrame({
            "age": [30, 35, 40, 45],
            "genre": ["M", "F", "M", "F"],
            "revenu_mensuel": [4000, 5000, 6000, 4500],
            "statut_marital": ["Marié(e)", "Célibataire", "Marié(e)", "Célibataire"],
            "departement": ["IT", "RH", "Ventes", "IT"],
            "poste": ["Dev", "Manager", "Manager", "Dev"],
            "nombre_experiences_precedentes": [2, 3, 5, 2],
            "annees_dans_l_entreprise": [3, 5, 7, 2],
            "note_evaluation_precedente": [4, 5, 3, 4],
            "heure_supplementaires": ["Oui", "Non", "Oui", "Non"],
            "augementation_salaire_precedente": ["15 %", "20 %", "10 %", "15 %"],
            "nombre_participation_pee": [2, 3, 1, 2],
            "nb_formations_suivies": [3, 4, 2, 3],
            "distance_domicile_travail": [10, 15, 5, 10],
            "niveau_education": [3, 4, 5, 3],
            "domaine_etude": ["Commerce", "Gestion", "IT", "Commerce"],
            "frequence_deplacement": ["Frequent", "Occasionnel", "Aucun", "Frequent"],
            "annees_depuis_la_derniere_promotion": [1, 2, 0, 1],
            "niveau_hierarchique_poste": [2, 3, 4, 2],
            "annee_experience_totale": [8, 10, 15, 7],
            "satisfaction_globale": [3.5, 4.0, 3.0, 3.5],
            "dispersion_satisfaction": [0.5, 0.3, 0.7, 0.5],
            "ratio_fidelite": [0.4, 0.6, 0.3, 0.4],
            "ratio_stagnation_poste": [0.5, 0.2, 0.8, 0.5],
            "duree_moyenne_experience": [2.0, 2.5, 3.0, 2.0],
            "salaire_par_annee_experience": [500.0, 500.0, 400.0, 500.0],
            "salaire_vs_poste": [1.0, 1.1, 0.9, 1.0],
            "salaire_vs_niveau": [1.0, 1.0, 1.1, 1.0]
        })
        y_train = pd.Series([0, 1, 0, 1])
        
        pipeline.fit(X_train, y_train)
        
        # Évaluer
        metrics = handler.evaluate_model(X_train, y_train)
        
        # Vérifier que les métriques sont présentes
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_predict(self):
        """Test prédiction avec la méthode predict"""
        handler = ModelHandler(model=LogisticRegression(max_iter=1000))
        pipeline = handler.build_pipeline()
        
        X_train = pd.DataFrame({
            "age": [30, 35],
            "genre": ["M", "F"],
            "revenu_mensuel": [4000, 5000],
            "statut_marital": ["Marié(e)", "Célibataire"],
            "departement": ["IT", "RH"],
            "poste": ["Dev", "Manager"],
            "nombre_experiences_precedentes": [2, 3],
            "annees_dans_l_entreprise": [3, 5],
            "note_evaluation_precedente": [4, 5],
            "heure_supplementaires": ["Oui", "Non"],
            "augementation_salaire_precedente": ["15 %", "20 %"],
            "nombre_participation_pee": [2, 3],
            "nb_formations_suivies": [3, 4],
            "distance_domicile_travail": [10, 15],
            "niveau_education": [3, 4],
            "domaine_etude": ["Commerce", "Gestion"],
            "frequence_deplacement": ["Frequent", "Aucun"],
            "annees_depuis_la_derniere_promotion": [1, 2],
            "niveau_hierarchique_poste": [2, 3],
            "annee_experience_totale": [8, 10],
            "satisfaction_globale": [3.5, 4.0],
            "dispersion_satisfaction": [0.5, 0.3],
            "ratio_fidelite": [0.4, 0.6],
            "ratio_stagnation_poste": [0.5, 0.2],
            "duree_moyenne_experience": [2.0, 2.5],
            "salaire_par_annee_experience": [500.0, 500.0],
            "salaire_vs_poste": [1.0, 1.1],
            "salaire_vs_niveau": [1.0, 1.0]
        })
        y_train = pd.Series([0, 1])
        
        pipeline.fit(X_train, y_train)
        
        predictions = handler.predict(X_train)
        assert predictions is not None
        assert len(predictions) == 2
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_proba(self):
        """Test prédiction probabiliste"""
        handler = ModelHandler(model=LogisticRegression(max_iter=1000))
        pipeline = handler.build_pipeline()
        
        X_train = pd.DataFrame({
            "age": [30, 35],
            "genre": ["M", "F"],
            "revenu_mensuel": [4000, 5000],
            "statut_marital": ["Marié(e)", "Célibataire"],
            "departement": ["IT", "RH"],
            "poste": ["Dev", "Manager"],
            "nombre_experiences_precedentes": [2, 3],
            "annees_dans_l_entreprise": [3, 5],
            "note_evaluation_precedente": [4, 5],
            "heure_supplementaires": ["Oui", "Non"],
            "augementation_salaire_precedente": ["15 %", "20 %"],
            "nombre_participation_pee": [2, 3],
            "nb_formations_suivies": [3, 4],
            "distance_domicile_travail": [10, 15],
            "niveau_education": [3, 4],
            "domaine_etude": ["Commerce", "Gestion"],
            "frequence_deplacement": ["Frequent", "Aucun"],
            "annees_depuis_la_derniere_promotion": [1, 2],
            "niveau_hierarchique_poste": [2, 3],
            "annee_experience_totale": [8, 10],
            "satisfaction_globale": [3.5, 4.0],
            "dispersion_satisfaction": [0.5, 0.3],
            "ratio_fidelite": [0.4, 0.6],
            "ratio_stagnation_poste": [0.5, 0.2],
            "duree_moyenne_experience": [2.0, 2.5],
            "salaire_par_annee_experience": [500.0, 500.0],
            "salaire_vs_poste": [1.0, 1.1],
            "salaire_vs_niveau": [1.0, 1.0]
        })
        y_train = pd.Series([0, 1])
        
        pipeline.fit(X_train, y_train)
        
        proba = handler.predict_proba(X_train)
        assert proba is not None
        assert len(proba) == 2


class TestModelSerialization:
    """Tests du chargement du modèle"""
    
    def test_load_model(self):
        """Test chargement du modèle sauvegardé"""
        model = load_model()
        # Peut être None si le modèle n'existe pas
        assert model is None or hasattr(model, "predict")
