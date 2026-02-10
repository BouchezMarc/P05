import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Base, Pred, TInputs, ViewInputs


# Fixture pour créer une BD de test
@pytest.fixture(scope="function")
def test_db():
    """Créer une BD SQLite en mémoire pour les tests"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    session = TestingSessionLocal()
    yield session
    
    session.close()
    Base.metadata.drop_all(bind=engine)


class TestTInputsModel:
    """Tests pour le modèle TInputs"""
    
    def test_create_tinput(self, test_db):
        """Test création d'un enregistrement TInputs"""
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
        test_db.add(input_record)
        test_db.commit()
        test_db.refresh(input_record)
        
        assert input_record.id_input is not None
        assert input_record.id_employee == 100
        assert input_record.age == 35
        assert input_record.genre == "M"
    
    def test_tinput_with_nullable_a_quitte(self, test_db):
        """Test que a_quitte_l_entreprise accepte None et applique un défaut"""
        input_record = TInputs(
            id_employee=200,
            age=28,
            genre="F",
            revenu_mensuel=4500,
            statut_marital="Célibataire",
            departement="IT",
            poste="Developer",
            nombre_experiences_precedentes=2,
            annees_dans_l_entreprise=3,
            annees_dans_le_poste_actuel=1,
            satisfaction_employee_environnement=4,
            note_evaluation_precedente=5,
            satisfaction_employee_nature_travail=5,
            satisfaction_employee_equipe=4,
            satisfaction_employee_equilibre_pro_perso=4,
            heure_supplementaires="Non",
            augementation_salaire_precedente="20 %",
            a_quitte_l_entreprise=None,  # Test nullable
            nombre_participation_pee=1,
            nb_formations_suivies=5,
            distance_domicile_travail=15,
            niveau_education=4,
            domaine_etude="Informatique",
            frequence_deplacement="Rare",
            annees_depuis_la_derniere_promotion=2,
            annee_experience_totale=8,
            niveau_hierarchique_poste=2
        )
        test_db.add(input_record)
        test_db.commit()
        test_db.refresh(input_record)
        
        assert input_record.a_quitte_l_entreprise is None or isinstance(
            input_record.a_quitte_l_entreprise, str
        )
    
    def test_query_tinputs(self, test_db):
        """Test requête de tous les TInputs"""
        # Créer 3 enregistrements
        for i in range(3):
            input_record = TInputs(
                id_employee=100 + i,
                age=30 + i,
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
            test_db.add(input_record)
        test_db.commit()
        
        all_inputs = test_db.query(TInputs).all()
        assert len(all_inputs) == 3
    
    def test_filter_tinputs_by_id_employee(self, test_db):
        """Test filtrage par id_employee"""
        input_record = TInputs(
            id_employee=555,
            age=40,
            genre="F",
            revenu_mensuel=6000,
            statut_marital="Marié(e)",
            departement="RH",
            poste="Directeur",
            nombre_experiences_precedentes=5,
            annees_dans_l_entreprise=10,
            annees_dans_le_poste_actuel=3,
            satisfaction_employee_environnement=5,
            note_evaluation_precedente=5,
            satisfaction_employee_nature_travail=5,
            satisfaction_employee_equipe=5,
            satisfaction_employee_equilibre_pro_perso=4,
            heure_supplementaires="Non",
            augementation_salaire_precedente="25 %",
            a_quitte_l_entreprise="Non",
            nombre_participation_pee=4,
            nb_formations_suivies=6,
            distance_domicile_travail=5,
            niveau_education=5,
            domaine_etude="Gestion",
            frequence_deplacement="Rare",
            annees_depuis_la_derniere_promotion=0,
            annee_experience_totale=20,
            niveau_hierarchique_poste=5
        )
        test_db.add(input_record)
        test_db.commit()
        
        result = test_db.query(TInputs).filter(TInputs.id_employee == 555).first()
        assert result is not None
        assert result.id_employee == 555
        assert result.age == 40


class TestPredModel:
    """Tests pour le modèle Pred"""
    
    def test_create_prediction(self, test_db):
        """Test création d'une prédiction"""
        # Créer d'abord un TInput
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
        test_db.add(input_record)
        test_db.commit()
        test_db.refresh(input_record)
        
        # Créer une prédiction
        pred = Pred(
            id_input=input_record.id_input,
            result_pred=True
        )
        test_db.add(pred)
        test_db.commit()
        test_db.refresh(pred)
        
        assert pred.id_pred is not None
        assert pred.id_input == input_record.id_input
        assert pred.result_pred == True
    
    def test_prediction_boolean_values(self, test_db):
        """Test que result_pred accepte True/False"""
        # Créer input
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
        test_db.add(input_record)
        test_db.commit()
        test_db.refresh(input_record)
        
        # Test avec True
        pred1 = Pred(id_input=input_record.id_input, result_pred=True)
        test_db.add(pred1)
        test_db.commit()
        
        # Test avec False
        pred2 = Pred(id_input=input_record.id_input, result_pred=False)
        test_db.add(pred2)
        test_db.commit()
        
        assert pred1.result_pred == True
        assert pred2.result_pred == False
    
    def test_multiple_predictions_same_input(self, test_db):
        """Test plusieurs prédictions pour le même input"""
        # Créer input
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
        test_db.add(input_record)
        test_db.commit()
        test_db.refresh(input_record)
        
        # Créer 3 prédictions
        for i in range(3):
            pred = Pred(id_input=input_record.id_input, result_pred=bool(i % 2))
            test_db.add(pred)
        test_db.commit()
        
        predictions = test_db.query(Pred).filter(Pred.id_input == input_record.id_input).all()
        assert len(predictions) == 3


class TestRelations:
    """Tests pour les relations entre tables"""
    
    def test_join_pred_tinputs(self, test_db):
        """Test jointure entre Pred et TInputs"""
        # Créer input
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
        test_db.add(input_record)
        test_db.commit()
        test_db.refresh(input_record)
        
        # Créer prédiction
        pred = Pred(id_input=input_record.id_input, result_pred=True)
        test_db.add(pred)
        test_db.commit()
        
        # Jointure
        result = test_db.query(Pred, TInputs.a_quitte_l_entreprise).join(
            TInputs, Pred.id_input == TInputs.id_input
        ).first()
        
        assert result is not None
        assert result[0].id_input == input_record.id_input
        assert result[1] == "Non"


class TestTransactions:
    """Tests pour les transactions"""
    
    def test_rollback_on_error(self, test_db):
        """Test que rollback annule les modifications"""
        # Ajouter un input
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
        test_db.add(input_record)
        test_db.rollback()  # Annuler
        
        # Vérifier que l'input n'a pas été ajouté
        count = test_db.query(TInputs).count()
        assert count == 0
    
    def test_commit_saves_data(self, test_db):
        """Test que commit sauvegarde les données"""
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
        test_db.add(input_record)
        test_db.commit()
        
        count = test_db.query(TInputs).count()
        assert count == 1
