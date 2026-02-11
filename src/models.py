from sqlalchemy import Column, Integer, String, Float, Boolean
from sqlalchemy.sql import func

from .bdd import Base


# -----------------------------------------------------------------


class Eval(Base):
    """
    Table eval contenent les données du csv eval
    """
    __tablename__ = "eval"

    eval_number = Column(String(10), primary_key=True, index=True)
    satisfaction_employee_environnement = Column(
        Integer, nullable=False, index=True
    )
    note_evaluation_precedente = Column(Integer, nullable=False)
    niveau_hierarchique_poste = Column(Integer, nullable=False)
    satisfaction_employee_nature_travail = Column(Integer, nullable=False)
    satisfaction_employee_equipe = Column(Integer, nullable=False)
    satisfaction_employee_equilibre_pro_perso = Column(Integer, nullable=False)
    note_evaluation_actuelle = Column(Integer, nullable=False)
    heure_supplementaires = Column(String(5), nullable=False)
    augementation_salaire_precedente = Column(
        String(5), primary_key=True, index=True
    )

    EXCLUDED_COLUMNS = {}

    def __repr__(self):
        values = ", ".join(
            f"{c.name}={getattr(self, c.name)!r}"
            for c in self.__table__.columns
            if c.name not in self.EXCLUDED_COLUMNS
        )
        return f"<{self.__class__.__name__}({values})>"

# -----------------------------------------------------------------


class Sirh(Base):

    """
    Table sirh contenent les données du csv sirh
    """
    __tablename__ = "sirh"

    id_employee = Column(Integer, primary_key=True, index=True)
    age = Column(Integer, nullable=False, index=True)
    genre = Column(String, nullable=False)
    revenu_mensuel = Column(Float, nullable=False)
    statut_marital = Column(String(15), nullable=False)
    departement = Column(String(50), nullable=False)
    poste = Column(String(50), nullable=False)
    nombre_experiences_precedentes = Column(Integer, nullable=False)
    nombre_heures_travailless = Column(Integer, nullable=False)
    annee_experience_totale = Column(Integer, nullable=False)
    annees_dans_l_entreprise = Column(Integer, nullable=False)
    annees_dans_le_poste_actuel = Column(Integer, nullable=False)

    EXCLUDED_COLUMNS = {}

    def __repr__(self):
        values = ", ".join(
            f"{c.name}={getattr(self, c.name)!r}"
            for c in self.__table__.columns
            if c.name not in self.EXCLUDED_COLUMNS
        )
        return f"<{self.__class__.__name__}({values})>"


# -----------------------------------------------------------------


class Sondage(Base):
    """
    Table sondage contenent les données du csv sondage
    """
    __tablename__ = "sondage"

    code_sondage = Column(Integer, primary_key=True, index=True)
    a_quitte_l_entreprise = Column(String(5), nullable=False)
    nombre_participation_pee = Column(Integer, nullable=False)
    nb_formations_suivies = Column(String(15), nullable=False)
    nombre_employee_sous_responsabilite = Column(String(50), nullable=False)
    distance_domicile_travail = Column(String(50), nullable=False)
    niveau_education = Column(Integer, nullable=False)
    domaine_etude = Column(String(50), nullable=False)
    ayant_enfants = Column(String(5), nullable=False)
    frequence_deplacement = Column(String(50), nullable=False)
    annees_depuis_la_derniere_promotion = Column(Integer, nullable=False)
    annes_sous_responsable_actuel = Column(Integer, nullable=False)

    EXCLUDED_COLUMNS = {}

    def __repr__(self):
        values = ", ".join(
            f"{c.name}={getattr(self, c.name)!r}"
            for c in self.__table__.columns
            if c.name not in self.EXCLUDED_COLUMNS
        )
        return f"<{self.__class__.__name__}({values})>"


# -----------------------------------------------------------------


class ViewRh(Base):
    """
    View basée sur les tables sirh, eval, sondage
    """
    __tablename__ = "view_rh"

    id_employee = Column(Integer, primary_key=True, index=True)
    age = Column(Integer, nullable=False)
    genre = Column(String, nullable=False)
    revenu_mensuel = Column(Float, nullable=False)
    statut_marital = Column(String(15), nullable=False)
    departement = Column(String(50), nullable=False)
    poste = Column(String(50), nullable=False)
    nombre_experiences_precedentes = Column(Integer, nullable=False)
    annees_dans_l_entreprise = Column(Integer, nullable=False)
    satisfaction_employee_environnement = Column(Integer, nullable=False)
    note_evaluation_precedente = Column(Integer, nullable=False)
    satisfaction_employee_nature_travail = Column(Integer, nullable=False)
    satisfaction_employee_equipe = Column(Integer, nullable=False)
    satisfaction_employee_equilibre_pro_perso = Column(Integer, nullable=False)
    heure_supplementaires = Column(Integer, nullable=False)
    augementation_salaire_precedente = Column(Integer, nullable=False)
    a_quitte_l_entreprise = Column(String, nullable=False)
    nombre_participation_pee = Column(Integer, nullable=False)
    nb_formations_suivies = Column(Integer, nullable=False)
    distance_domicile_travail = Column(Integer, nullable=False)
    niveau_education = Column(Integer, nullable=False)
    domaine_etude = Column(String, nullable=False)
    frequence_deplacement = Column(String, nullable=False)
    annees_depuis_la_derniere_promotion = Column(Integer, nullable=False)
    niveau_hierarchique_poste = Column(Integer, nullable=False)
    annees_dans_le_poste_actuel = Column(Integer, nullable=False)
    annee_experience_totale = Column(Integer, nullable=False)
    satisfaction_globale = Column(Float, nullable=False)
    dispersion_satisfaction = Column(Float, nullable=False)
    ratio_fidelite = Column(Float, nullable=False)
    ratio_stagnation_poste = Column(Float, nullable=False)
    duree_moyenne_experience = Column(Float, nullable=False)
    salaire_par_annee_experience = Column(Float, nullable=False)
    salaire_vs_poste = Column(Float, nullable=False)
    salaire_vs_niveau = Column(Float, nullable=False)

    EXCLUDED_COLUMNS = {
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

    def __repr__(self):
        values = ", ".join(
            f"{c.name}={getattr(self, c.name)!r}"
            for c in self.__table__.columns
            if c.name not in self.EXCLUDED_COLUMNS
        )
        return f"<{self.__class__.__name__}({values})>"


# -----------------------------------------------------------------


class Pred(Base):
    """
    Table contenant les prédictions
    """
    __tablename__ = "prediction"

    id_pred = Column(Integer, primary_key=True, index=True)
    id_input = Column(Integer, nullable=False)
    # Boolean car la colonne DB est de type boolean.
    result_pred = Column(Boolean, nullable=False)

    EXCLUDED_COLUMNS = {}

    def __repr__(self):
        values = ", ".join(
            f"{c.name}={getattr(self, c.name)!r}"
            for c in self.__table__.columns
            if c.name not in self.EXCLUDED_COLUMNS
        )
        return f"<{self.__class__.__name__}({values})>"


# -----------------------------------------------------------------


class TInputs(Base):
    """
    Table contenant les infos sur les inputs et le dataset de test après split
    """
    __tablename__ = "inputs"

    id_input = Column(Integer, primary_key=True, index=True)
    id_employee = Column(Integer, nullable=True)
    age = Column(Integer, nullable=False)
    genre = Column(String, nullable=False, server_default=func.now())
    revenu_mensuel = Column(Integer, nullable=False)
    statut_marital = Column(String, nullable=False, server_default=func.now())
    departement = Column(String, nullable=False, server_default=func.now())
    poste = Column(String, nullable=False, server_default=func.now())
    nombre_experiences_precedentes = Column(Integer, nullable=False)
    annees_dans_l_entreprise = Column(Integer, nullable=False)
    annees_dans_le_poste_actuel = Column(Integer, nullable=False)
    satisfaction_employee_environnement = Column(Integer, nullable=False)
    note_evaluation_precedente = Column(Integer, nullable=False)
    satisfaction_employee_nature_travail = Column(Integer, nullable=False)
    satisfaction_employee_equipe = Column(Integer, nullable=False)
    satisfaction_employee_equilibre_pro_perso = Column(Integer, nullable=False)
    heure_supplementaires = Column(
        String, nullable=False, server_default=func.now()
    )
    augementation_salaire_precedente = Column(
        String, nullable=False, server_default=func.now()
    )
    a_quitte_l_entreprise = Column(
        String, nullable=True, server_default=func.now()
    )
    nombre_participation_pee = Column(Integer, nullable=False)
    nb_formations_suivies = Column(Integer, nullable=False)
    distance_domicile_travail = Column(Integer, nullable=False)
    niveau_education = Column(Integer, nullable=False)
    domaine_etude = Column(
        String, nullable=False, server_default=func.now()
    )
    frequence_deplacement = Column(
        String, nullable=False, server_default=func.now()
    )
    annees_depuis_la_derniere_promotion = Column(Integer, nullable=False)
    annee_experience_totale = Column(Integer, nullable=False)
    # nombre_heures_travailless = Column(Integer, nullable=False)
    # nombre_employee_s_responsabilite = Column(Integer, nullable=False)
    # ayant_enfants = Column(String, nullable=False, server_default=func.now())
    # code_sondage = Column(Integer, nullable=False)
    # eval_number = Column(String, nullable=False, server_default=func.now())
    # note_evaluation_actuelle = Column(Integer, nullable=False)
    # annes_s_responsable_actuel = Column(Integer, nullable=False)
    niveau_hierarchique_poste = Column(Integer, nullable=False)

    EXCLUDED_COLUMNS = {}

    def __repr__(self):
        values = ", ".join(
            f"{c.name}={getattr(self, c.name)!r}"
            for c in self.__table__.columns
            if c.name not in self.EXCLUDED_COLUMNS
        )
        return f"<{self.__class__.__name__}({values})>"


class ViewInputs(Base):
    """
    View basée sur la table inputs
    """
    __tablename__ = "view_inputs"
    id_input = Column(Integer, primary_key=True, index=True)
    id_employee = Column(Integer, nullable=False)
    age = Column(Integer, nullable=False)
    genre = Column(String, nullable=False)
    revenu_mensuel = Column(Float, nullable=False)
    statut_marital = Column(String(15), nullable=False)
    departement = Column(String(50), nullable=False)
    poste = Column(String(50), nullable=False)
    nombre_experiences_precedentes = Column(Integer, nullable=False)
    annees_dans_l_entreprise = Column(Integer, nullable=True)
    satisfaction_employee_environnement = Column(Integer, nullable=False)
    note_evaluation_precedente = Column(Integer, nullable=False)
    satisfaction_employee_nature_travail = Column(Integer, nullable=False)
    satisfaction_employee_equipe = Column(Integer, nullable=False)
    satisfaction_employee_equilibre_pro_perso = Column(Integer, nullable=False)
    heure_supplementaires = Column(Integer, nullable=False)
    augementation_salaire_precedente = Column(Integer, nullable=False)
    a_quitte_l_entreprise = Column(String, nullable=False)
    nombre_participation_pee = Column(Integer, nullable=False)
    nb_formations_suivies = Column(Integer, nullable=False)
    distance_domicile_travail = Column(Integer, nullable=False)
    niveau_education = Column(Integer, nullable=False)
    domaine_etude = Column(String, nullable=False)
    frequence_deplacement = Column(String, nullable=False)
    annees_depuis_la_derniere_promotion = Column(Integer, nullable=False)
    niveau_hierarchique_poste = Column(Integer, nullable=False)
    annees_dans_le_poste_actuel = Column(Integer, nullable=False)
    annee_experience_totale = Column(Integer, nullable=False)
    satisfaction_globale = Column(Float, nullable=False)
    dispersion_satisfaction = Column(Float, nullable=False)
    ratio_fidelite = Column(Float, nullable=False)
    ratio_stagnation_poste = Column(Float, nullable=False)
    duree_moyenne_experience = Column(Float, nullable=False)
    salaire_par_annee_experience = Column(Float, nullable=False)
    salaire_vs_poste = Column(Float, nullable=False)
    salaire_vs_niveau = Column(Float, nullable=False)

    EXCLUDED_COLUMNS = {
        # "nombre_heures_travailless",
        # "nombre_employee_sous_responsabilite",
        # "ayant_enfants",
        # "annees_dans_le_poste_actuel",
        # "note_evaluation_actuelle",
        # "code_sondage",
        # "eval_number",
        # "satisfaction_employee_nature_travail",
        # "satisfaction_employee_equipe",
        # "satisfaction_employee_equilibre_pro_perso",
        # "satisfaction_employee_environnement",
    }

    def __repr__(self):
        values = ", ".join(
            f"{c.name}={getattr(self, c.name)!r}"
            for c in self.__table__.columns
            if c.name not in self.EXCLUDED_COLUMNS
        )
        return f"<{self.__class__.__name__}({values})>"
