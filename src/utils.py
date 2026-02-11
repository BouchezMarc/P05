import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .models import ViewRh, TInputs
from .bdd import SessionLocal


# All Utils Functions
print("utils")

# -------------------------------------------------------
# Files Paths

CREATE_TABLE_SQL_PATH = "./sql/create_tables_p5_rh.sql"
# CREATE_VIEW = "./sql/p5_view_rh.sql"
INSERT_FILES = {
    "sirh": "./sql/extrait_sirh_insert.csv",
    "eval": "./sql/extrait_eval_insert.csv",
    "sondage": "./sql/extrait_sondage_insert.csv",
}
TABLES = ["sirh", "eval", "sondage"]

# -------------------------------------------------------
# Query from Text files


def execute_sql_file(conn, file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        sql = f.read()
        conn.execute(text(sql))
    conn.commit()


# -------------------------------------------------------
# Select pour vérifier si la table contient du data
def table_is_empty(db, table_name):
    result = db.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
    # Récupérer la première ligne du résultat avec fetchone()
    count = result.fetchone()[0]

    # Retourner si la table est vide (count == 0)
    return count == 0


# =========================
# Init de Base des Tables
# =========================
def create_bd_base():
    # print("🔌 Test connexion à la base...")
    db = SessionLocal()
    db.autocommit = False
    print(CREATE_TABLE_SQL_PATH)
    try:

        # 1. Test table
        execute_sql_file(db, CREATE_TABLE_SQL_PATH)

        # 2. Remplissage si vide
        for table in TABLES:
            print(table)

            if table_is_empty(db, table):
                print(f"📥 Table '{table}' vide → insertion CSV")

                file_path = INSERT_FILES[table]
                print(f"➡️ Fichier utilisé : {file_path}")

                execute_sql_file(db, file_path)

            else:
                print(f"📊 Table '{table}' déjà remplie")

    except Exception as e:
        db.rollback()
        print("❌ Erreur :", e)

    finally:
        db.close()


# -------------------------------------------------------
# Lecture du Dataset ------------------------------------
def query_init_df():
    try:
        # Get session
        db = SessionLocal()

        try:
            # Exécuter la requête pour récupérer
            # toutes les lignes de la table ViewRh
            rows = db.query(ViewRh).all()

            # Vérifier si des données ont été récupérées
            if not rows:
                raise ValueError("❌ Aucune donnée dans la BDD")

            # Convertir les résultats en DataFrame Pandas
            df = pd.DataFrame([r.__dict__ for r in rows])
            # Remove SQLAlchemy internal column if present
            df = df.drop(columns=["_sa_instance_state"], errors="ignore")

        finally:
            db.close()
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        db.close()

    return df


# -------------------------------------------------------
# Ecriture du Dataset après le split

# Fonction pour insérer un DataFrame dans la table log_split
def insert_data_into_db(df_test: pd.DataFrame, db):
    inputs = []

    # Conserver uniquement les colonnes reconnues par le modèle LogSplit
    allowed_cols = set(TInputs.__table__.columns.keys())
    required_cols = {
        c.name
        for c in TInputs.__table__.columns
        if not c.nullable
        and c.default is None
        and c.server_default is None
        and c.name not in {"id_input"}
    }
    print(f"🔍 Colonnes autorisées pour Inputs: {sorted(allowed_cols)}")
    print(
        f"🔍 Colonnes df entrantes: {sorted(df_test.columns.tolist())}"
    )

    missing_required = sorted(required_cols - set(df_test.columns))
    if missing_required:
        print(f"[ERROR] Missing required columns: {missing_required}")
        raise ValueError(
            "Colonnes requises manquantes pour LogSplit: "
            f"{missing_required}"
        )

    # Parcourir chaque ligne du DataFrame
    for _, row in df_test.iterrows():
        inputs_data = {}

        # Copier uniquement les colonnes attendues par LogSplit
        for col in df_test.columns:
            if col in allowed_cols:
                inputs_data[col] = row[col]

        # Ajouter l'id du modèle dans chaque entrée
        # log_split_data["id_ml"] = id_ml

        # Créer un objet LogSplit avec les données dynamiques
        input = TInputs(**inputs_data)

        # Ajouter l'objet à la liste
        inputs.append(input)

    # print(f"[INFO] Ready to insert {len(inputs)} rows")
    if inputs:
        sample = inputs[0].__dict__.copy()
        sample.pop("_sa_instance_state", None)
        # print(f"[DEBUG] Sample payload: {sample}")

    # Insérer toutes les lignes dans la base de données
    try:
        db.add_all(inputs)
        db.flush()  # force la remontée des erreurs éventuelles avant commit
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Flush error: {e}")
        raise


# Fonction principale pour insérer les données
def insert_train_data(df_test: pd.DataFrame):
    db = SessionLocal()
    try:
        # Insérer les données d'entraînement dans log_split
        insert_data_into_db(df_test, db)
        db.commit()
        count_after = db.query(TInputs).count()
        print(f"[SUCCESS] inputs now has {count_after} rows")
        print("[SUCCESS] Training data inserted!")
    except SQLAlchemyError as e:
        db.rollback()
        print(f"[ERROR] Insert data error: {e}")
    finally:
        db.close()
