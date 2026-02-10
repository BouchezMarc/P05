from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from pathlib import Path
import joblib

from .utils import create_bd_base, query_init_df, insert_train_data


# Transformations -----------------------------------------------------------

binary_map = {
    'genre': {'F': 0, 'M': 1},
    'heure_supplementaires': {'Non': 0, 'Oui': 1},
    # 'a_quitte_l_entreprise': {'Non': 0, 'Oui': 1}
}


def transform_binary(X):
    return X.replace(binary_map)


def transform_percent(X):
    return X.iloc[:, 0].str.replace(' %', '', regex=False).astype(int).to_frame()


def transform_freq(X):
    return X.replace({'Aucun': 0, 'Occasionnel': 1, 'Frequent': 2}).infer_objects()

binary_transformer = FunctionTransformer(
    transform_binary,
    feature_names_out='one-to-one'
)

percent_transformer = FunctionTransformer(
    transform_percent,
    feature_names_out='one-to-one'
)

freq_transformer = FunctionTransformer(
    transform_freq,
    feature_names_out='one-to-one'
)
# Colums ---------------------------------------------------------

cat_cols = ['statut_marital', 'departement','poste', 'domaine_etude']
binary_cols = ['genre', 'heure_supplementaires'] # ,'a_quitte_l_entreprise']
percent_col = ['augementation_salaire_precedente']
freq_col = ['frequence_deplacement']

preprocessor = ColumnTransformer(
    transformers=[
        ('binary', binary_transformer, binary_cols),
        ('percent', percent_transformer, percent_col),
        ('freq', freq_transformer, freq_col),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
    ],
    #remainder='drop'
    remainder='passthrough'
)

# Class Pipeline --------------------------------------------------


class ModelHandler:
    def __init__(self, model, stratify=True, preprocessor=None, n_splits=5):
        """
        :param model: Modèle de machine learning (e.g., LogisticRegression)

        :param stratify: Si True, utilise une stratification dans la validation croisée
        :param preprocessor: Transformations de prétraitement (e.g., StandardScaler, OneHotEncoder)
        :param n_splits: Nombre de splits dans la validation croisée
        """
        self.model = model
        self.stratify = stratify
        self.n_splits = n_splits  # Nombre de folds pour la validation croisée

        # Colonnes par défaut (issues du module)
        self.cat_cols = cat_cols
        self.binary_cols = binary_cols
        self.percent_col = percent_col
        self.freq_col = freq_col

        self.preprocessor = preprocessor
        

    @staticmethod
    def transform_binary(X):
        binary_map = {'genre': {'F': 0, 'M': 1}, 'heure_supplementaires': {'Non': 0, 'Oui': 1},}
        return X.replace(binary_map)    
    @staticmethod
    def transform_percent(X):
        return X.iloc[:, 0].str.replace(' %', '', regex=False).astype(int).to_frame()
    @staticmethod
    def transform_freq(X):
        return X.replace({'Aucun': 0, 'Occasionnel': 1, 'Frequent': 2}).infer_objects()

    def _build_preprocessor(self):
        binary_transformer = FunctionTransformer(
            self.transform_binary,
            feature_names_out='one-to-one'
        )
        percent_transformer = FunctionTransformer(
            self.transform_percent,
            feature_names_out='one-to-one'
        )
        freq_transformer = FunctionTransformer(
            self.transform_freq,
            feature_names_out='one-to-one'
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ('binary', binary_transformer, self.binary_cols),
                ('percent', percent_transformer, self.percent_col),
                ('freq', freq_transformer, self.freq_col),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), self.cat_cols),
            ],
            remainder='passthrough'
        )
        return preprocessor



    def build_pipeline(self, X_train=None):
            """ Build the pipeline with preprocessing, and the model """
            steps = []
            # Add preprocessor if it is defined, otherwise build it
            if self.preprocessor is None:
                self.preprocessor = self._build_preprocessor()
            steps.append(('prep', self.preprocessor))

            # Add the model
            steps.append(('model', self.model))
            # Create the pipeline
            self.pipeline = Pipeline(steps)
            return self.pipeline


    def train_model(self, X_train, y_train):
        """ Train the model using StratifiedKFold """
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)       

        # Perform StratifiedKFold
        for train_idx, val_idx in skf.split(X_train, y_train):
            # Utilisation de iloc pour indexer X_train et y_train correctement
            X_train_fold = X_train.iloc[train_idx]  # .iloc pour DataFrame
            # X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]  # .iloc pour Series
            #y_val_fold = y_train.iloc[val_idx]

            # Fit the model on the current fold
            self.pipeline.fit(X_train_fold, y_train_fold)

    def evaluate_model(self, X_test, y_test):
        """ Évaluer le modèle sur l'ensemble de test """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        # Prédictions
        y_pred = self.pipeline.predict(X_test)
        
        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
        
        return metrics

    def predict(self, X):
        """ Faire des prédictions avec le modèle """
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """ Faire des prédictions probabilistes """
        return self.pipeline.predict_proba(X)    

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# Trainning model

if __name__ == "__main__":
    print("launched directly train.py")

    # ---------------
    create_bd_base()

    # Import Dataframe-------------------------------------

    df = query_init_df()

    df['a_quitte_l_entreprise'] = df['a_quitte_l_entreprise'].apply({'Non': 0, 'Oui': 1}.get)
    # print (df)

    y = df['a_quitte_l_entreprise']
    id_employee = df['id_employee']
    X = df.drop(columns=['a_quitte_l_entreprise', 'id_employee'])

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    df_test = X_test.copy()

    df_test['a_quitte_l_entreprise'] = y_test
    df_test['id_employee'] = id_employee.loc[X_test.index]

    df_train = X_train.copy()

    df_train['a_quitte_l_entreprise'] = y_train
    df_train['id_employee'] = id_employee.loc[X_train.index]

    # insert du dataframe en BDD
    print("##################################")
    print("insert")
    print(df_test)
    insert_train_data(df_test)

    # Supprimer les colonnes exclues pour l'entraînement du modèle
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
 
    X_train = X_train.drop(columns=[col for col in excluded_cols if col in X_train.columns], errors='ignore')
   
    
    # Param Models ----------------------------------

    model = LogisticRegression(
        #fbeta=2     
        C=0.1,     
        class_weight={0: 1, 1: 10},     
        max_iter=1000,     
        penalty='l2',     
        solver='newton-cg'   
    )
    # Param Pipeline --------------------------------

    handler = ModelHandler(
        model=model,    
        preprocessor=preprocessor
    )

    # Trainning -------------------------------

    handler.build_pipeline(X_train)
    handler.train_model(X_train, y_train)

    # Save trainning --------------------------
    try:
        ARTIFACT_PATH = Path(__file__).parent.parent / "model" / "ml_model.joblib"
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(handler.pipeline, ARTIFACT_PATH)
        print(f"✅ Modèle sauvegardé: {ARTIFACT_PATH}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du modèle: {e}")
    # -----------------------------------------------   

else:
    print("launched indirectly")
