-- Table: public.eval

-- DROP TABLE IF EXISTS public.eval;

CREATE TABLE IF NOT EXISTS public.eval
(
    satisfaction_employee_environnement integer,
    note_evaluation_precedente integer,
    niveau_hierarchique_poste integer,
    satisfaction_employee_nature_travail integer,
    satisfaction_employee_equipe integer,
    satisfaction_employee_equilibre_pro_perso integer,
    note_evaluation_actuelle integer,
    heure_supplementaires character varying(5) COLLATE pg_catalog."default",
    augementation_salaire_precedente character varying(5) COLLATE pg_catalog."default",
    eval_number character varying(10) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT eval_pkey PRIMARY KEY (eval_number)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.eval
    OWNER to postgres;
GRANT SELECT ON public.eval TO authenticated;

-- Table: public.sirh

-- DROP TABLE IF EXISTS public.sirh;

CREATE TABLE IF NOT EXISTS public.sirh
(
    id_employee integer NOT NULL,
    age integer,
    genre "char",
    revenu_mensuel integer,
    statut_marital character varying(15) COLLATE pg_catalog."default",
    departement character varying(50) COLLATE pg_catalog."default",
    poste character varying(50) COLLATE pg_catalog."default",
    nombre_experiences_precedentes integer,
    nombre_heures_travailless integer,
    annee_experience_totale integer,
    annees_dans_l_entreprise integer,
    annees_dans_le_poste_actuel integer,
    CONSTRAINT sirh_pkey PRIMARY KEY (id_employee)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.sirh
    OWNER to postgres;

GRANT SELECT ON public.sirh TO authenticated;
-- Table: public.sondage

-- DROP TABLE IF EXISTS public.sondage;

CREATE TABLE IF NOT EXISTS public.sondage
(
    a_quitte_l_entreprise character varying(5) COLLATE pg_catalog."default",
    nombre_participation_pee integer,
    nb_formations_suivies integer,
    nombre_employee_sous_responsabilite integer,
    code_sondage integer NOT NULL,
    distance_domicile_travail integer,
    niveau_education integer,
    domaine_etude character varying(50) COLLATE pg_catalog."default",
    ayant_enfants character varying(5) COLLATE pg_catalog."default",
    frequence_deplacement character varying(50) COLLATE pg_catalog."default",
    annees_depuis_la_derniere_promotion integer,
    annes_sous_responsable_actuel integer,
    CONSTRAINT sondage_pkey PRIMARY KEY (code_sondage)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.sondage
    OWNER to postgres;	

GRANT SELECT ON public.sondage TO authenticated;
---------------------------------------------------------------------------

-- Table: public.prediction

DROP TABLE IF EXISTS public.prediction;

CREATE TABLE IF NOT EXISTS public.prediction
(
    id_pred integer NOT NULL GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    id_input integer NOT NULL,
    result_pred boolean NOT NULL
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.prediction
    OWNER to postgres;

COMMENT ON TABLE public.prediction
    IS 'log les prédictions du ML';

GRANT SELECT ON public.prediction TO authenticated;	
---------------------------------------------------------------------------
DROP VIEW IF EXISTS public.view_inputs;
--DROP TABLE IF EXISTS public.inputs;

CREATE TABLE IF NOT EXISTS public.inputs
(
    id_input integer NOT NULL GENERATED ALWAYS AS IDENTITY PRIMARY KEY,    
    id_employee integer ,
    age integer NOT NULL,
    genre character varying(1),
    revenu_mensuel integer NOT NULL,
    statut_marital character varying(15) COLLATE pg_catalog."default",
    departement character varying(50) COLLATE pg_catalog."default",
    poste character varying(50) COLLATE pg_catalog."default",
    nombre_experiences_precedentes integer NOT NULL,
    annees_dans_l_entreprise integer NOT NULL,
    annees_dans_le_poste_actuel integer NOT NULL,
    satisfaction_employee_environnement integer NOT NULL,
    note_evaluation_precedente integer NOT NULL,
    satisfaction_employee_nature_travail integer NOT NULL,
    satisfaction_employee_equipe integer NOT NULL,
    satisfaction_employee_equilibre_pro_perso integer NOT NULL,
    heure_supplementaires character varying(5) COLLATE pg_catalog."default",
    augementation_salaire_precedente character varying(5) COLLATE pg_catalog."default",
    a_quitte_l_entreprise character varying(5) COLLATE pg_catalog."default",
    nombre_participation_pee integer NOT NULL,
    nb_formations_suivies integer NOT NULL,
    distance_domicile_travail integer NOT NULL,
    niveau_education integer NOT NULL,
    domaine_etude character varying(50) COLLATE pg_catalog."default",
    frequence_deplacement character varying(50) COLLATE pg_catalog."default",
    annees_depuis_la_derniere_promotion integer NOT NULL,    
    annee_experience_totale integer NOT NULL,   
	niveau_hierarchique_poste integer NOT NULL
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.inputs
    OWNER to postgres;

COMMENT ON TABLE public.inputs
    IS 'log les id des employés de test lors du split et lors des inputs';
	

GRANT SELECT ON public.inputs TO authenticated;	
-------------------------------------------------
DROP VIEW IF EXISTS view_rh;
CREATE VIEW view_rh AS
WITH joined AS (
    SELECT          
			s.id_employee,
            s.age,
            s.genre,
            s.revenu_mensuel,
            s.statut_marital,
            s.departement,
            s.poste,
            s.nombre_experiences_precedentes,            
            s.annees_dans_l_entreprise,
            e.satisfaction_employee_environnement,
            e.note_evaluation_precedente,
            e.satisfaction_employee_nature_travail,
            e.satisfaction_employee_equipe,
            e.satisfaction_employee_equilibre_pro_perso,
            e.heure_supplementaires,
            e.augementation_salaire_precedente,            
            so.a_quitte_l_entreprise,
            so.nombre_participation_pee,
            so.nb_formations_suivies,            
            so.distance_domicile_travail,
            so.niveau_education,
            so.domaine_etude,
            so.frequence_deplacement,
            so.annees_depuis_la_derniere_promotion,			 
			-- constantes
			s.nombre_heures_travailless,
            so.nombre_employee_sous_responsabilite,
			so.ayant_enfants,
			-- id			
			so.code_sondage,	
			e.eval_number,
			-- correlations trops fortes
			e.note_evaluation_actuelle,
			so.annes_sous_responsable_actuel,	
			e.niveau_hierarchique_poste,
			s.annees_dans_le_poste_actuel,
			s.annee_experience_totale

    FROM sirh s
    JOIN eval e
        ON s.id_employee = CAST(SUBSTRING(e.eval_number, 3) AS INT)
    JOIN sondage so
        ON s.id_employee = so.code_sondage
)

SELECT
    *,
    (satisfaction_employee_environnement +
     satisfaction_employee_nature_travail +
     satisfaction_employee_equipe +
     satisfaction_employee_equilibre_pro_perso) / 4.0
        AS satisfaction_globale,

SQRT(
    (
        POWER(satisfaction_employee_environnement, 2) +
        POWER(satisfaction_employee_nature_travail, 2) +
        POWER(satisfaction_employee_equipe, 2) +
        POWER(satisfaction_employee_equilibre_pro_perso, 2)
    ) / 4.0
    -
    POWER(
        (
            satisfaction_employee_environnement +
            satisfaction_employee_nature_travail +
            satisfaction_employee_equipe +
            satisfaction_employee_equilibre_pro_perso
        ) / 4.0,
        2
    )
) AS dispersion_satisfaction,

	COALESCE(
	    annees_dans_l_entreprise / NULLIF(annee_experience_totale, 0),0
	) AS ratio_fidelite,

	COALESCE(
    annees_dans_le_poste_actuel / NULLIF(annees_dans_l_entreprise, 0),0
        )AS ratio_stagnation_poste,
	
    annee_experience_totale / (nombre_experiences_precedentes + 1)
        AS duree_moyenne_experience,

    revenu_mensuel / (annee_experience_totale + 1)
        AS salaire_par_annee_experience,

    revenu_mensuel / AVG(revenu_mensuel) OVER (PARTITION BY poste)
        AS salaire_vs_poste,

    revenu_mensuel / AVG(revenu_mensuel) OVER (PARTITION BY niveau_hierarchique_poste)
        AS salaire_vs_niveau

FROM joined;
GRANT SELECT ON public.view_rh TO authenticated;

---------------------------------------------------------------------------

-- View: public.view_inputs



CREATE VIEW public.view_inputs
 AS
 WITH base AS (
         SELECT 
			inputs.id_input,
			inputs.id_employee,
            inputs.age,
            inputs.genre,
            inputs.revenu_mensuel,
            inputs.statut_marital,
            inputs.departement,
            inputs.poste,
            inputs.nombre_experiences_precedentes,
            inputs.annees_dans_l_entreprise,
            inputs.satisfaction_employee_environnement,
            inputs.note_evaluation_precedente,
            inputs.satisfaction_employee_nature_travail,
            inputs.satisfaction_employee_equipe,
            inputs.satisfaction_employee_equilibre_pro_perso,
            inputs.heure_supplementaires,
            inputs.augementation_salaire_precedente,
            inputs.a_quitte_l_entreprise,
            inputs.nombre_participation_pee,
            inputs.nb_formations_suivies,
            inputs.distance_domicile_travail,
            inputs.niveau_education,
            inputs.domaine_etude,
            inputs.frequence_deplacement,
            inputs.annees_depuis_la_derniere_promotion,
            inputs.annees_dans_le_poste_actuel,
            inputs.annee_experience_totale,
            inputs.niveau_hierarchique_poste
           FROM inputs
        )
 SELECT 
	id_input,
	id_employee,
    age,
    genre,
    revenu_mensuel,
    statut_marital,
    departement,
    poste,
    nombre_experiences_precedentes,
    annees_dans_l_entreprise,
    satisfaction_employee_environnement,
    note_evaluation_precedente,
    satisfaction_employee_nature_travail,
    satisfaction_employee_equipe,
    satisfaction_employee_equilibre_pro_perso,
    heure_supplementaires,
    augementation_salaire_precedente,
    a_quitte_l_entreprise,
    nombre_participation_pee,
    nb_formations_suivies,
    distance_domicile_travail,
    niveau_education,
    domaine_etude,
    frequence_deplacement,
    annees_depuis_la_derniere_promotion,
    annees_dans_le_poste_actuel,
    annee_experience_totale,
    niveau_hierarchique_poste,
    (satisfaction_employee_environnement + satisfaction_employee_nature_travail + satisfaction_employee_equipe + satisfaction_employee_equilibre_pro_perso)::numeric / 4.0 AS satisfaction_globale,
    sqrt((power(satisfaction_employee_environnement::double precision, 2::double precision) + power(satisfaction_employee_nature_travail::double precision, 2::double precision) + power(satisfaction_employee_equipe::double precision, 2::double precision) + power(satisfaction_employee_equilibre_pro_perso::double precision, 2::double precision)) / 4.0::double precision - power((satisfaction_employee_environnement + satisfaction_employee_nature_travail + satisfaction_employee_equipe + satisfaction_employee_equilibre_pro_perso)::numeric / 4.0, 2::numeric)::double precision) AS dispersion_satisfaction,
    COALESCE(annees_dans_l_entreprise / NULLIF(annee_experience_totale, 0), 0) AS ratio_fidelite,
    COALESCE(annees_dans_le_poste_actuel / NULLIF(annees_dans_l_entreprise, 0), 0) AS ratio_stagnation_poste,
    annee_experience_totale / (nombre_experiences_precedentes + 1) AS duree_moyenne_experience,
    revenu_mensuel / (annee_experience_totale + 1) AS salaire_par_annee_experience,
    revenu_mensuel::numeric / avg(revenu_mensuel) OVER (PARTITION BY poste) AS salaire_vs_poste,
    revenu_mensuel::numeric / avg(revenu_mensuel) OVER (PARTITION BY niveau_hierarchique_poste) AS salaire_vs_niveau
   FROM base;

ALTER TABLE public.view_inputs
    OWNER TO postgres;
GRANT SELECT ON public.view_inputs TO authenticated;	

	

	
ALTER TABLE public.eval ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sirh ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sondage ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.inputs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.prediction ENABLE ROW LEVEL SECURITY;
--ALTER TABLE public.view_rh ENABLE ROW LEVEL SECURITY;
--ALTER TABLE public.view_inputs ENABLE ROW LEVEL SECURITY;

--REVOKE ALL ON public.view_rh FROM public;
--REVOKE ALL ON public.view_inputs FROM public;

--GRANT SELECT ON public.view_rh TO authenticated;
--GRANT SELECT ON public.view_inputs TO authenticated;