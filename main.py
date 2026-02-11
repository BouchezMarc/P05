from typing import List, Optional
import numpy as np
from fastapi import FastAPI, Depends
from pydantic import ConfigDict,BaseModel
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

from src.predict import predict, load_model
from src.train import transform_binary, transform_percent, transform_freq
from src.models import ViewInputs, Pred, TInputs
from src.bdd import SessionLocal
from src.utils import create_bd_base

# Variable globale pour le modèle
model = None

# Lifespan pour charger le modèle au démarrage
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model
    create_bd_base()
    model = load_model()
    print("[INFO] Modèle chargé avec succès au démarrage")
    yield
    # Shutdown (si besoin de nettoyage)
    print("[INFO] Arrêt de l'application")

# Définir FastAPI avec lifespan
app = FastAPI(lifespan=lifespan)

# Fonction pour obtenir une session de base de données
def get_db():
    db = SessionLocal()  # Utilisation de SessionLocal définie dans src/bdd.py
    try:
        yield db
    finally:
        db.close()

#-------------------------------------
# défault value de la table inputs

class InputInsertRequest(BaseModel):
    id_employee: int = 999
    age: int = 35
    genre: str = "M"
    revenu_mensuel: int = 5000
    statut_marital: str = "Marié(e)"
    departement: str = "Ventes"
    poste: str = "Manager"
    nombre_experiences_precedentes: int = 3
    annees_dans_l_entreprise: int = 5
    annees_dans_le_poste_actuel: int = 2
    satisfaction_employee_environnement: int = 3
    note_evaluation_precedente: int = 4
    satisfaction_employee_nature_travail: int = 4
    satisfaction_employee_equipe: int = 4
    satisfaction_employee_equilibre_pro_perso: int = 3
    heure_supplementaires: str = "Oui"
    augementation_salaire_precedente: str = "15 %"
    a_quitte_l_entreprise: Optional[str] = "Non"  # Optionnel
    nombre_participation_pee: int = 2
    nb_formations_suivies: int = 3
    distance_domicile_travail: int = 10
    niveau_education: int = 3
    domaine_etude: str = "Commerce"
    frequence_deplacement: str = "Frequent"
    annees_depuis_la_derniere_promotion: int = 1
    annee_experience_totale: int = 10
    niveau_hierarchique_poste: int = 3

#-------------------------------------
# Définition de la table inputs

class PredictionRequest(BaseModel):
    id_input: int
    age: int
    revenu_mensuel: float
    note_evaluation_precedente: int
    heure_supplementaires: int
    augementation_salaire_precedente: int
    a_quitte_l_entreprise: str
    nombre_participation_pee: int
    nb_formations_suivies: int
    distance_domicile_travail: int
    niveau_education: int
    domaine_etude: str
    frequence_deplacement: str
    annees_depuis_la_derniere_promotion: int
    niveau_hierarchique_poste: int
    annee_experience_totale: int
    satisfaction_globale: float
    dispersion_satisfaction: float
    ratio_fidelite: float
    ratio_stagnation_poste: float
    duree_moyenne_experience: float
    salaire_par_annee_experience: float
    salaire_vs_poste: float
    salaire_vs_niveau: float

#-------------------------------------

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

#-------------------------------------

# Endpoint pour vérifier la santé de l'API et du modèle
@app.get("/health")
def health_check():
    if model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "message": "Le modèle n'est pas chargé"
        }
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "message": "API et modèle fonctionnels"
    }

#-------------------------------------
# inputs

# Endpoint pour récupérer toutes les données de la table inputs
@app.get("/inputs")
def get_all_inputs(db: Session = Depends(get_db)):
    try:
        # Récupérer toutes les données de la table inputs
        inputs = db.query(TInputs).all()
        
        if not inputs:
            return {
                "message": "Aucune donnée trouvée dans la table inputs",
                "total": 0,
                "inputs": []
            }
        
        # Formater les résultats
        inputs_list = []
        for input_record in inputs:
            inputs_list.append({
                "id_input": input_record.id_input,
                "id_employee": input_record.id_employee,
                "age": input_record.age,
                "genre": input_record.genre,
                "revenu_mensuel": input_record.revenu_mensuel,
                "statut_marital": input_record.statut_marital,
                "departement": input_record.departement,
                "poste": input_record.poste,
                "nombre_experiences_precedentes": input_record.nombre_experiences_precedentes,
                "annees_dans_l_entreprise": input_record.annees_dans_l_entreprise,
                "annees_dans_le_poste_actuel": input_record.annees_dans_le_poste_actuel,
                "satisfaction_employee_environnement": input_record.satisfaction_employee_environnement,
                "note_evaluation_precedente": input_record.note_evaluation_precedente,
                "satisfaction_employee_nature_travail": input_record.satisfaction_employee_nature_travail,
                "satisfaction_employee_equipe": input_record.satisfaction_employee_equipe,
                "satisfaction_employee_equilibre_pro_perso": input_record.satisfaction_employee_equilibre_pro_perso,
                "heure_supplementaires": input_record.heure_supplementaires,
                "augementation_salaire_precedente": input_record.augementation_salaire_precedente,
                "a_quitte_l_entreprise": input_record.a_quitte_l_entreprise,
                "nombre_participation_pee": input_record.nombre_participation_pee,
                "nb_formations_suivies": input_record.nb_formations_suivies,
                "distance_domicile_travail": input_record.distance_domicile_travail,
                "niveau_education": input_record.niveau_education,
                "domaine_etude": input_record.domaine_etude,
                "frequence_deplacement": input_record.frequence_deplacement,
                "annees_depuis_la_derniere_promotion": input_record.annees_depuis_la_derniere_promotion,
                "annee_experience_totale": input_record.annee_experience_totale,
                "niveau_hierarchique_poste": input_record.niveau_hierarchique_poste
            })
        
        return {
            "message": "Données récupérées avec succès",
            "total": len(inputs_list),
            "inputs": inputs_list
        }
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

#-------------------------------------
# /predict

# Endpoint predict all from table inputs
@app.get("/predict")
def make_prediction(db: Session = Depends(get_db)):
    try:
        # Récupérer les données de `view_inputs` depuis la base de données
        data = db.query(ViewInputs).all()
        print(f"[DEBUG] Nombre de lignes récupérées: {len(data)}")
        
        if not data:
            return {"error": "Aucune donnée trouvée dans ViewInputs"}

        # Préparer un DataFrame avec les colonnes réelles de ViewInputs
        import pandas as pd
        df_data = []
        id_inputs = []  # Pour stocker les id_input correspondants
        
        for row in data:
            df_data.append({
                'age': row.age,
                'genre': row.genre,
                'revenu_mensuel': row.revenu_mensuel,
                'statut_marital': row.statut_marital,
                'departement': row.departement,
                'poste': row.poste,
                'nombre_experiences_precedentes': row.nombre_experiences_precedentes,
                'annees_dans_l_entreprise': row.annees_dans_l_entreprise,
                'note_evaluation_precedente': row.note_evaluation_precedente,
                'heure_supplementaires': row.heure_supplementaires,
                'augementation_salaire_precedente': row.augementation_salaire_precedente,
                'nombre_participation_pee': row.nombre_participation_pee,
                'nb_formations_suivies': row.nb_formations_suivies,
                'distance_domicile_travail': row.distance_domicile_travail,
                'niveau_education': row.niveau_education,
                'domaine_etude': row.domaine_etude,
                'frequence_deplacement': row.frequence_deplacement,
                'annees_depuis_la_derniere_promotion': row.annees_depuis_la_derniere_promotion,
                'niveau_hierarchique_poste': row.niveau_hierarchique_poste,
                'annee_experience_totale': row.annee_experience_totale,
                'satisfaction_globale': row.satisfaction_globale,
                'dispersion_satisfaction': row.dispersion_satisfaction,
                'ratio_fidelite': row.ratio_fidelite,
                'ratio_stagnation_poste': row.ratio_stagnation_poste,
                'duree_moyenne_experience': row.duree_moyenne_experience,
                'salaire_par_annee_experience': row.salaire_par_annee_experience,
                'salaire_vs_poste': row.salaire_vs_poste,
                'salaire_vs_niveau': row.salaire_vs_niveau,
            })
            id_inputs.append(row.id_input)
        
        df = pd.DataFrame(df_data)
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        
        # Supprimer les colonnes exclues si elles existent
        excluded_cols = {
            "nombre_heures_travailless",
            "nombre_employee_sous_responsabilite",
            "ayant_enfants",
            "annees_dans_le_poste_actuel",
            "note_evaluation_actuelle",        
            "code_sondage",
            "eval_number",
            "satisfaction_employee_nature_travail",
            "satisfaction_employee_equipe",
            "satisfaction_employee_equilibre_pro_perso",
            "satisfaction_employee_environnement",
        }
        df = df.drop(columns=[col for col in excluded_cols if col in df.columns], errors='ignore')
        print(f"[DEBUG] DataFrame shape après suppression colonnes exclues: {df.shape}")
        
        # Nettoyer les NaN
        print(f"[DEBUG] NaN avant nettoyage: {df.isna().sum().sum()}")
        df = df.dropna()
        print(f"[DEBUG] DataFrame shape après dropna: {df.shape}")
        print(f"[DEBUG] NaN après nettoyage: {df.isna().sum().sum()}")
        
        if df.empty:
            return {"error": "Aucune donnée valide après nettoyage (toutes les lignes avaient des NaN)"}

        # Faire les prédictions avec la pipeline complète
        predictions = model.predict(df).tolist()
        print(f"[DEBUG] Prédictions: {predictions}")
        
        # Compter les 0 et 1
        count_0 = predictions.count(0)
        count_1 = predictions.count(1)
        
        # Sauvegarder les prédictions dans la table prediction
        prediction_results = []
        for id_input, prediction in zip(id_inputs, predictions):
            pred = Pred(
                id_input=id_input,
                result_pred=bool(prediction)  # Convertir en booléen
            )
            db.add(pred)
            prediction_results.append({
                "id_input": id_input,
                "prediction": int(prediction)
            })
        
        # Valider les modifications en base de données
        db.commit()
        print(f"[INFO] {len(prediction_results)} prédictions sauvegardées dans la table prediction")
        
        return {
            "message": f"{len(prediction_results)} prédictions sauvegardées",
            "total": len(prediction_results),
            "count_non": count_0,
            "count_oui": count_1,
            "predictions": prediction_results
        }
    
    except Exception as e:
        db.rollback()
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

#-------------------------------------
# /history

# Endpoint pour récupérer l'historique des prédictions
@app.get("/history")
def get_prediction_history(db: Session = Depends(get_db)):
    try:
        # Récupérer toutes les prédictions avec une jointure sur la table inputs
        predictions = db.query(Pred, TInputs.a_quitte_l_entreprise).join(
            TInputs, Pred.id_input == TInputs.id_input
        ).all()
        
        if not predictions:
            return {
                "message": "Aucun historique de prédiction trouvé",
                "total": 0,
                "count_non": 0,
                "count_oui": 0,
                "history": []
            }
        
        # Formater les résultats
        history = []
        count_oui = 0
        count_non = 0
        
        for pred, a_quitte in predictions:
            result_int = 1 if pred.result_pred else 0
            if result_int == 1:
                count_oui += 1
            else:
                count_non += 1
                
            history.append({
                "id_pred": pred.id_pred,
                "id_input": pred.id_input,
                "result_pred": result_int,
                "a_quitte_l_entreprise": a_quitte
            })
        
        return {
            "message": "Historique récupéré avec succès",
            "total": len(history),
            "count_non": count_non,
            "count_oui": count_oui,
            "history": history
        }
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

#-------------------------------------
# /history_by_id

# Endpoint pour récupérer l'historique des prédictions d'un input spécifique
@app.get("/history_by_id/{id_input}")
def get_prediction_history_by_id(id_input: int, db: Session = Depends(get_db)):
    try:
        # Récupérer toutes les prédictions pour cet id_input avec la valeur réelle
        predictions = db.query(Pred, TInputs.a_quitte_l_entreprise).join(
            TInputs, Pred.id_input == TInputs.id_input
        ).filter(Pred.id_input == id_input).all()
        
        if not predictions:
            return {
                "message": f"Aucune prédiction trouvée pour id_input={id_input}",
                "id_input": id_input,
                "total": 0,
                "count_non": 0,
                "count_oui": 0,
                "history": []
            }
        
        # Formater les résultats
        history = []
        count_oui = 0
        count_non = 0
        a_quitte_value = None
        
        for pred, a_quitte in predictions:
            result_int = 1 if pred.result_pred else 0
            if result_int == 1:
                count_oui += 1
            else:
                count_non += 1
            
            a_quitte_value = a_quitte  # Garder la valeur réelle
                
            history.append({
                "id_pred": pred.id_pred,
                "result_pred": result_int,
                "result_label": "Oui" if result_int == 1 else "Non"
            })
        
        return {
            "message": f"Historique récupéré pour id_input={id_input}",
            "id_input": id_input,
            "a_quitte_l_entreprise": a_quitte_value,
            "total": len(history),
            "count_non": count_non,
            "count_oui": count_oui,
            "history": history
        }
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

#-------------------------------------
# /predict_by_id

# Endpoint pour faire une prédiction sur un input spécifique
@app.get("/predict_by_id/{id_input}")
def predict_by_id(id_input: int, db: Session = Depends(get_db)):
    try:
        # Récupérer les données de l'input spécifique
        input_data = db.query(ViewInputs).filter(ViewInputs.id_input == id_input).first()
        
        if not input_data:
            return {"error": f"Aucun input trouvé avec id_input={id_input}"}
        
        # Récupérer la valeur réelle a_quitte_l_entreprise depuis TInputs
        input_record = db.query(TInputs).filter(TInputs.id_input == id_input).first()
        a_quitte_value = input_record.a_quitte_l_entreprise if input_record else None
        
        # Préparer un DataFrame avec les données de cet input
        import pandas as pd
        df_data = {
            'age': input_data.age,
            'genre': input_data.genre,
            'revenu_mensuel': input_data.revenu_mensuel,
            'statut_marital': input_data.statut_marital,
            'departement': input_data.departement,
            'poste': input_data.poste,
            'nombre_experiences_precedentes': input_data.nombre_experiences_precedentes,
            'annees_dans_l_entreprise': input_data.annees_dans_l_entreprise,
            'note_evaluation_precedente': input_data.note_evaluation_precedente,
            'heure_supplementaires': input_data.heure_supplementaires,
            'augementation_salaire_precedente': input_data.augementation_salaire_precedente,
            'nombre_participation_pee': input_data.nombre_participation_pee,
            'nb_formations_suivies': input_data.nb_formations_suivies,
            'distance_domicile_travail': input_data.distance_domicile_travail,
            'niveau_education': input_data.niveau_education,
            'domaine_etude': input_data.domaine_etude,
            'frequence_deplacement': input_data.frequence_deplacement,
            'annees_depuis_la_derniere_promotion': input_data.annees_depuis_la_derniere_promotion,
            'niveau_hierarchique_poste': input_data.niveau_hierarchique_poste,
            'annee_experience_totale': input_data.annee_experience_totale,
            'satisfaction_globale': input_data.satisfaction_globale,
            'dispersion_satisfaction': input_data.dispersion_satisfaction,
            'ratio_fidelite': input_data.ratio_fidelite,
            'ratio_stagnation_poste': input_data.ratio_stagnation_poste,
            'duree_moyenne_experience': input_data.duree_moyenne_experience,
            'salaire_par_annee_experience': input_data.salaire_par_annee_experience,
            'salaire_vs_poste': input_data.salaire_vs_poste,
            'salaire_vs_niveau': input_data.salaire_vs_niveau,
        }
        
        df = pd.DataFrame([df_data])
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        
        # Supprimer les colonnes exclues si elles existent
        excluded_cols = {
            "nombre_heures_travailless",
            "nombre_employee_sous_responsabilite",
            "ayant_enfants",
            "annees_dans_le_poste_actuel",
            "note_evaluation_actuelle",        
            "code_sondage",
            "eval_number",
            "satisfaction_employee_nature_travail",
            "satisfaction_employee_equipe",
            "satisfaction_employee_equilibre_pro_perso",
            "satisfaction_employee_environnement",
        }
        df = df.drop(columns=[col for col in excluded_cols if col in df.columns], errors='ignore')
        
        # Faire la prédiction
        prediction = model.predict(df)[0]
        print(f"[DEBUG] Prédiction pour id_input={id_input}: {prediction}")
        
        # Sauvegarder la prédiction dans la table prediction
        pred = Pred(
            id_input=id_input,
            result_pred=bool(prediction)
        )
        db.add(pred)
        db.commit()
        
        print(f"[INFO] Prédiction sauvegardée pour id_input={id_input}")
        
        return {
            "message": "Prédiction effectuée avec succès",
            "id_input": id_input,
            "prediction": int(prediction),            
            "a_quitte_l_entreprise": a_quitte_value
        }
    
    except Exception as e:
        db.rollback()
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

#-------------------------------------
# /predict/insert

# Endpoint pour insérer un nouvel input et faire une prédiction
@app.post("/predict/insert")
def insert_and_predict(data: InputInsertRequest, db: Session = Depends(get_db)):
    try:
        # Créer un nouvel enregistrement dans la table inputs
        new_input = TInputs(
            id_employee=data.id_employee,
            age=data.age,
            genre=data.genre,
            revenu_mensuel=data.revenu_mensuel,
            statut_marital=data.statut_marital,
            departement=data.departement,
            poste=data.poste,
            nombre_experiences_precedentes=data.nombre_experiences_precedentes,
            annees_dans_l_entreprise=data.annees_dans_l_entreprise,
            annees_dans_le_poste_actuel=data.annees_dans_le_poste_actuel,
            satisfaction_employee_environnement=data.satisfaction_employee_environnement,
            note_evaluation_precedente=data.note_evaluation_precedente,
            satisfaction_employee_nature_travail=data.satisfaction_employee_nature_travail,
            satisfaction_employee_equipe=data.satisfaction_employee_equipe,
            satisfaction_employee_equilibre_pro_perso=data.satisfaction_employee_equilibre_pro_perso,
            heure_supplementaires=data.heure_supplementaires,
            augementation_salaire_precedente=data.augementation_salaire_precedente,
            a_quitte_l_entreprise=data.a_quitte_l_entreprise,  # Peut être None
            nombre_participation_pee=data.nombre_participation_pee,
            nb_formations_suivies=data.nb_formations_suivies,
            distance_domicile_travail=data.distance_domicile_travail,
            niveau_education=data.niveau_education,
            domaine_etude=data.domaine_etude,
            frequence_deplacement=data.frequence_deplacement,
            annees_depuis_la_derniere_promotion=data.annees_depuis_la_derniere_promotion,
            annee_experience_totale=data.annee_experience_totale,
            niveau_hierarchique_poste=data.niveau_hierarchique_poste
        )
        
        db.add(new_input)
        db.flush()  # Pour obtenir l'id_input généré
        
        id_input_created = new_input.id_input
        print(f"[INFO] Nouvel input créé avec id_input={id_input_created}")
        
        # Récupérer les données depuis view_inputs pour la prédiction
        input_data = db.query(ViewInputs).filter(ViewInputs.id_input == id_input_created).first()
        
        if not input_data:
            db.rollback()
            return {"error": f"Impossible de récupérer les données de ViewInputs pour id_input={id_input_created}"}
        
        # Préparer un DataFrame pour la prédiction
        import pandas as pd
        df_data = {
            'age': input_data.age,
            'genre': input_data.genre,
            'revenu_mensuel': input_data.revenu_mensuel,
            'statut_marital': input_data.statut_marital,
            'departement': input_data.departement,
            'poste': input_data.poste,
            'nombre_experiences_precedentes': input_data.nombre_experiences_precedentes,
            'annees_dans_l_entreprise': input_data.annees_dans_l_entreprise,
            'note_evaluation_precedente': input_data.note_evaluation_precedente,
            'heure_supplementaires': input_data.heure_supplementaires,
            'augementation_salaire_precedente': input_data.augementation_salaire_precedente,
            'nombre_participation_pee': input_data.nombre_participation_pee,
            'nb_formations_suivies': input_data.nb_formations_suivies,
            'distance_domicile_travail': input_data.distance_domicile_travail,
            'niveau_education': input_data.niveau_education,
            'domaine_etude': input_data.domaine_etude,
            'frequence_deplacement': input_data.frequence_deplacement,
            'annees_depuis_la_derniere_promotion': input_data.annees_depuis_la_derniere_promotion,
            'niveau_hierarchique_poste': input_data.niveau_hierarchique_poste,
            'annee_experience_totale': input_data.annee_experience_totale,
            'satisfaction_globale': input_data.satisfaction_globale,
            'dispersion_satisfaction': input_data.dispersion_satisfaction,
            'ratio_fidelite': input_data.ratio_fidelite,
            'ratio_stagnation_poste': input_data.ratio_stagnation_poste,
            'duree_moyenne_experience': input_data.duree_moyenne_experience,
            'salaire_par_annee_experience': input_data.salaire_par_annee_experience,
            'salaire_vs_poste': input_data.salaire_vs_poste,
            'salaire_vs_niveau': input_data.salaire_vs_niveau,
        }
        
        df = pd.DataFrame([df_data])
        
        # Supprimer les colonnes exclues
        excluded_cols = {
            "nombre_heures_travailless",
            "nombre_employee_sous_responsabilite",
            "ayant_enfants",
            "annees_dans_le_poste_actuel",
            "note_evaluation_actuelle",
            "code_sondage",
            "eval_number",
            "satisfaction_employee_nature_travail",
            "satisfaction_employee_equipe",
            "satisfaction_employee_equilibre_pro_perso",
            "satisfaction_employee_environnement",
        }
        df = df.drop(columns=[col for col in excluded_cols if col in df.columns], errors='ignore')
        
        # Faire la prédiction
        prediction = model.predict(df)[0]
        print(f"[DEBUG] Prédiction pour le nouvel input: {prediction}")
        
        # Sauvegarder la prédiction
        pred = Pred(
            id_input=id_input_created,
            result_pred=bool(prediction)
        )
        db.add(pred)
        db.commit()
        
        print(f"[INFO] Input et prédiction sauvegardés avec succès")
        
        return {
            "message": "Input créé et prédiction effectuée avec succès",
            "id_input": id_input_created,
            "id_employee": data.id_employee,
            "prediction": int(prediction),
            #"prediction_label": "Oui" if prediction == 1 else "Non"
        }
    
    except Exception as e:
        db.rollback()
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    
