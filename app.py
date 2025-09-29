import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
from pathlib import Path
from joblib import load
from PIL import Image
from matplotlib.ticker import MultipleLocator

# Configuration du logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Initialisation de l'état de session
if 'run_predictions' not in st.session_state:
    st.session_state.run_predictions = False

# Charger ton logo
logo_1 = Image.open(r"C:\Users\pc\Desktop\nolithhor_logo.png")

# Config de la page avec ton logo comme icône
st.set_page_config(
    page_title="LithoVision Pro",
    page_icon=logo_1,   # <-- ici ton logo
    layout="wide",
    initial_sidebar_state="expanded"
)

# Nouvelle palette adaptée au logo
theme = {
    "primary": "#009688",   # Turquoise (cube)
    "secondary": "#0D1B26", # Sidebar bleu nuit
    "background": "#1b3a4b",# Fond clair
    "text": "#333333",      # Texte foncé lisible
    "accent": "#FCC13E"     # Jaune doré (symbole central)
}

# ----- Barre horizontale (option 1: supprimer totalement) -----
# Réafficher la barre horizontale (header) avec fond vert uni
st.markdown("""
<style>
header[data-testid="stHeader"] {
    background: #009688 !important;  /* Vert turquoise uni */
    color: white !important;
    height: 45px;
}
header[data-testid="stHeader"] .stDeployButton {display: none;}  /* cacher "Deploy" */
header[data-testid="stHeader"] button[kind="header"] {display: none;}  /* cacher menu ... */
</style>
""", unsafe_allow_html=True)


# ----- CSS amélioré -----
st.markdown(f"""
<style>
    .stApp {{
        background: {theme['background']};
        color: {theme['text']};
        font-family: 'Segoe UI', sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background: {theme['secondary']} !important;
        color: {theme['accent']} !important;
        box-shadow: 5px 0 15px rgba(0,0,0,0.3);
    }}
    h1, h2, h3 {{
        color: {theme['primary']} !important;
        font-weight: 600;
    }}
    .stButton>button {{
        background: {theme['accent']} !important;
        color: #1C1C1C !important;
        border-radius: 10px;
        font-weight: bold;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background: #FFD54F !important;
        transform: scale(1.05);
    }}
    .stFileUploader>div>div {{
        border: 2px dashed {theme['primary']} !important;
        border-radius: 10px;
        padding: 18px;
        background: rgba(0, 150, 136, 0.05);
    }}
    label, .stTextInput>div>div>input {{
        color: {theme['accent']} !important;
    }}
    h2, h3, .stTextInput>div>div>input {{
        color: {theme['primary']} !important;
    }}
    h1, .stTextInput>div>div>input {{
        color: {theme['accent']} !important;
    }}


    
</style>
""", unsafe_allow_html=True)



# Configuration des couleurs de lithologie
LITHO_COLORS = {
    "VCL": "#3E4743",
    "Quartz": "#FDFE02",
    "Igneous": "#DE6AE5",
    "PIGE": "#C8C8C8",
    "Calcite": "#87CEEB",
    "Anhydrite": "#800080",
    "Dolomite": "#205C7E",
    "Halite": "#AAAAFF",
    "Autres_litho": "#FFFFFF00"
}

@st.cache_resource
def load_models():
    """Charge les modèles pré-entraînés"""
    try:
        base_path = "C:/Users/pc/Desktop/lithology_app/src/models/trained"
        models = {
            "VCL_reg": load(f'{base_path}/Regression/reg_VCL.joblib'),
            "Quartz_reg": load(f'{base_path}/Regression/reg_Quartz.joblib'),
            "Igneous_cls": load(f'{base_path}/Classification/cls_Igneous.joblib'),
            "Igneous_reg": load(f'{base_path}/Regression/reg_Igneous.joblib'),
            "Autres_cls": load(f'{base_path}/Classification/cls_Autres_litho.joblib'),
            "Autres_reg": load(f'{base_path}/Regression/reg_Autres_litho.joblib'),
            "post_cls": load(f'{base_path}/Classification/post_classification.joblib'),
            "PIGE_reg": load(f'{base_path}/Regression/reg_PIGE.joblib'),
        }
        logger.info("Modèles chargés avec succès")
        return models
    except Exception as e:
        logger.error(f"Erreur de chargement des modèles: {str(e)}")
        st.error("Erreur de chargement des modèles - Voir les logs")
        return None

def process_file(uploaded_file):
    """Charge et prétraite le fichier"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Format de fichier non supporté. Utilisez CSV ou XLSX.")
            return None
    except Exception as e:
        logger.error(f"Erreur de traitement du fichier: {str(e)}")
        st.error("Erreur de lecture du fichier")
        return None

def run_predictions(df, models, VCL_cutoff=0.40):
    """Exécute les prédictions de lithologie"""
    try:
        # Remplir les NaN avec 0 pour éviter les erreurs de prédiction
        df.fillna(0, inplace=True)

        # ----------------------- VCL -----------------------
        features_vcl = ['CALX', 'GR', 'CNC', 'KTH', 'DTCQI', 'K', 'TH', 'ZDEN']
        df['VCL_pred'] = models['VCL_reg'].predict(df[features_vcl])

        # ----------------------- Quartz -----------------------
        features_quartz = ['CALX', 'GR', 'CNC', 'DTCQI', 'K', 'KTH', 'TH', 'ZDEN']
        df['Quartz_pred'] = models['Quartz_reg'].predict(df[features_quartz])

        # ----------------------- Igneous -----------------------
        features_igneous_cls = ['CALX', 'CNC', 'DTCQI', 'GR', 'KTH', 'K', 'PE', 'TH', 'U', 'ZDEN']
        df["Igneous_cls"] = models["Igneous_cls"].predict(df[features_igneous_cls])

        features_igneous_reg = ['CALX', 'GR', 'CNC', 'DTCQI', 'K', 'KTH', 'TH', 'ZDEN']
        mask_igneous = df["Igneous_cls"] == 1
        df["Igneous_reg"] = 0.0
        if mask_igneous.any():
            df.loc[mask_igneous, "Igneous_reg"] = models["Igneous_reg"].predict(
                df.loc[mask_igneous, features_igneous_reg]
            )

        # ----------------------- Autres lithologies -----------------------
        features_autres_cls = ['CALX', 'CNC', 'DTCQI', 'GR', 'KTH', 'K', 'PE', 'TH', 'U', 'ZDEN']
        df["Autres_cls"] = models["Autres_cls"].predict(df[features_autres_cls])
        logger.info(f"Autres_cls - Valeurs uniques: {df['Autres_cls'].unique()}")
        logger.info(f"Autres_cls - Nombre de 1: {sum(df['Autres_cls'] == 1)}")

        features_autres_reg = ['CALX', 'GR', 'CNC', 'DTCQI', 'K', 'KTH', 'TH', 'ZDEN']
        mask_autres = df["Autres_cls"] == 1
        df["Autres_reg"] = 0.0
        if mask_autres.any():
            df.loc[mask_autres, "Autres_reg"] = models["Autres_reg"].predict(
                df.loc[mask_autres, features_autres_reg]
            )
        logger.info(f"Autres_reg - Stats: min={df['Autres_reg'].min()}, max={df['Autres_reg'].max()}, >0: {sum(df['Autres_reg'] > 0)}")

        # ----------------------- Post-classification -----------------------
        features_post = ['CALX', 'CNC', 'DTCQI', 'GR', 'KTH', 'K', 'PE', 'TH', 'U', 'ZDEN']
        mask_autres = df['Autres_cls'] == 1
        
        # Initialiser la colonne Final_lithology
        df['Final_lithology'] = np.nan
        
        # Prédire seulement pour les lignes avec Autres_cls == 1
        if mask_autres.any():
            df.loc[mask_autres, 'Final_lithology'] = models['post_cls'].predict(
                df.loc[mask_autres, features_post]
            )

        logger.info(f"Final_lithology - Valeurs uniques: {df['Final_lithology'].unique()}")
        logger.info(f"Final_lithology - Counts: {df['Final_lithology'].value_counts()}")    

        # ----------------- Initialisation garantie des colonnes autres lithologies -----------------
        for col in ['_Calcite_prop', '_Anhydrite_prop', '_Dolomite_prop', '_Halite_prop']:
            if col not in df.columns:
                df[col] = 0.0

        # Dictionnaire mapping int → nom
        int_to_name = {0: "Dolomite", 1: "Anhydrite", 2: "Halite", 3: "Calcite"}

        # Masque lignes "autres" - pour débogage, on peut temporairement enlever la condition Autres_reg > 0
        au_mask = (df['Autres_cls'] == 1)  # & (df['Autres_reg'] > 0)  # <-- décommentez la condition plus tard

        logger.info(f"Nombre de lignes sélectionnées par au_mask: {au_mask.sum()}")
        logger.info(f"Autres_reg stats: min={df['Autres_reg'].min()}, max={df['Autres_reg'].max()}")
        logger.info(f"Final_lithology unique values (au_mask): {df.loc[au_mask, 'Final_lithology'].unique()}")
        logger.info(f"Final_lithology types (au_mask): {df.loc[au_mask, 'Final_lithology'].apply(type).unique()}")

        if 'Autres_reg' not in df.columns:
            df['Autres_reg'] = 0.0

        # Remplir les colonnes des minéraux
        for idx in df[au_mask].index:
            autres_val = df.at[idx, 'Autres_reg']
            pred = df.at[idx, 'Final_lithology']

            if pd.isna(pred):
                continue

            pred_name = None
            if isinstance(pred, (int, np.integer, float, np.floating)):
                pred_name = int_to_name.get(int(pred))
            elif isinstance(pred, str):
                pred_lower = pred.lower()
                if 'calcite' in pred_lower:
                    pred_name = "Calcite"
                elif 'anhydrite' in pred_lower:
                    pred_name = "Anhydrite"
                elif 'dolomite' in pred_lower:
                    pred_name = "Dolomite"
                elif 'halite' in pred_lower:
                    pred_name = "Halite"

            if pred_name == "Calcite":
                df.at[idx, '_Calcite_prop'] = autres_val
            elif pred_name == "Anhydrite":
                df.at[idx, '_Anhydrite_prop'] = autres_val
            elif pred_name == "Dolomite":
                df.at[idx, '_Dolomite_prop'] = autres_val
            elif pred_name == "Halite":
                df.at[idx, '_Halite_prop'] = autres_val

        # Journaliser pour le débogage
        logger.info(f"Calcite total: {df['_Calcite_prop'].sum()}")
        logger.info(f"Anhydrite total: {df['_Anhydrite_prop'].sum()}")
        logger.info(f"Dolomite total: {df['_Dolomite_prop'].sum()}")
        logger.info(f"Halite total: {df['_Halite_prop'].sum()}")

        # # Afficher les informations de débogage dans l'interface
        # st.write(f"Nombre de lignes avec Autres_cls = 1: {sum(df['Autres_cls'] == 1)}")
        # st.write(f"Autres_reg > 0: {sum(df['Autres_reg'] > 0)}")
        # st.write(f"Final_lithology valeurs uniques: {df['Final_lithology'].dropna().unique()}")
        # st.write(f"Total Calcite: {df['_Calcite_prop'].sum()}")
        # st.write(f"Total Anhydrite: {df['_Anhydrite_prop'].sum()}")
        # st.write(f"Total Dolomite: {df['_Dolomite_prop'].sum()}")
        # st.write(f"Total Halite: {df['_Halite_prop'].sum()}")

        # ----------------------- PIGE -----------------------
        features_pige = ['CALX', 'GR', 'CNC', 'DTCQI', 'K', 'KTH', 'TH', 'ZDEN', 'VCL_pred']
        df['PIGE_pred'] = models['PIGE_reg'].predict(df[features_pige])

        # Correction règles géologiques
        df["PIGE_final"] = df["PIGE_pred"]
        df.loc[df["VCL_pred"] > VCL_cutoff, "PIGE_final"] = 0

        # Appliquer la nouvelle logique
        # if "PIGE_final" in df.columns and "Quartz_pred" in df.columns:
        #     mask_condition = (df["PIGE_final"] > 0.02) & (df["Quartz_pred"] > 0.7)
        #     if mask_condition.any():
        #         df.loc[mask_condition, "Igneous_cls"] = 0
        #         df.loc[mask_condition, "_Dolomite_prop"] = 0
        #         df.loc[mask_condition, "_Calcite_prop"] = 0
        #         df.loc[mask_condition, "_Anhydrite_prop"] = 0
        #         df.loc[mask_condition, "_Halite_prop"] = 0

        #     mask_pige_zero = (df["PIGE_final"] <= 0.02)
        #     if mask_pige_zero.any():
        #         df.loc[mask_pige_zero, "PIGE_final"] = 0   

        return df

    except Exception as e:
        logger.error(f"Erreur lors des prédictions: {str(e)}")
        st.error(f"Erreur de prédiction: {str(e)}")
        return None

    


def plot_stacked(results: pd.DataFrame):
    """Affiche le graphique empilé des lithologies"""
    df = results.copy()

    # Vérification et création des colonnes manquantes
    for col in ['_Calcite_prop', '_Anhydrite_prop', '_Dolomite_prop', '_Halite_prop']:
        if col not in df.columns:
            df[col] = 0.0
            st.warning(f"La colonne {col} était absente et a été créée avec des zéros.")
        elif df[col].sum() == 0:
            st.warning(f"La colonne {col} contient uniquement des zéros.")

    # Vérification des colonnes requises
    required_columns = ['DEPTH', 'VCL_pred', 'Quartz_pred', 'Igneous_reg', 'PIGE_final']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Colonnes manquantes dans les données : {', '.join(missing_columns)}")
        return

    # Préparer les données brutes
    VCL_raw = df.get('VCL_pred', pd.Series(0.0, index=df.index)).clip(lower=0)
    Q_raw = df.get('Quartz_pred', pd.Series(0.0, index=df.index)).clip(lower=0)
    I_raw = df.get('Igneous_reg', pd.Series(0.0, index=df.index)).clip(lower=0)
    P_raw = df.get('PIGE_final', pd.Series(0.0, index=df.index)).clip(lower=0)
    C_raw = df['_Calcite_prop'].clip(lower=0)
    A_raw = df['_Anhydrite_prop'].clip(lower=0)
    D_raw = df['_Dolomite_prop'].clip(lower=0)
    H_raw = df['_Halite_prop'].clip(lower=0)

    # Normalisation
    non_vcl_sum = Q_raw + I_raw + C_raw + A_raw + D_raw + H_raw + P_raw
    target_non_vcl = (1.0 - VCL_raw).clip(lower=0)
    scale = pd.Series(0.0, index=df.index)
    nz = non_vcl_sum > 0
    scale.loc[nz] = target_non_vcl.loc[nz] / non_vcl_sum.loc[nz]
    scale.fillna(0, inplace=True)

    Q = Q_raw * scale
    I = I_raw * scale
    C = C_raw * scale
    A = A_raw * scale
    D = D_raw * scale
    H = H_raw * scale
    P = P_raw * scale
    V = VCL_raw

    # Vérification des bornes
    for s in (Q, I, C, A, D, H, P, V):
        s.clip(lower=0, upper=1, inplace=True)

    # Tri des profondeurs
    depth = df['DEPTH'].values
    order = np.argsort(depth)
    depth = depth[order]
    V = V.values[order]; Q = Q.values[order]; I = I.values[order]
    C = C.values[order]; A = A.values[order]; D = D.values[order]; H = H.values[order]; P = P.values[order]

    # Calcul des limites
    left_V = np.zeros_like(V)
    right_V = V

    left_Q = right_V
    right_Q = left_Q + Q

    left_I = right_Q
    right_I = left_I + I

    left_C = right_I
    right_C = left_C + C

    left_A = right_C
    right_A = left_A + A

    left_D = right_A
    right_D = left_D + D

    left_H = right_D
    right_H = left_H + H

    left_P = right_H
    right_P = left_P + P

    fig, ax = plt.subplots(figsize=(1.5, 60))

    lithologies = [
        ("VCL", left_V, right_V),
        ("Quartz", left_Q, right_Q),
        ("Igneous", left_I, right_I),
        ("Calcite", left_C, right_C),
        ("Anhydrite", left_A, right_A),
        ("Dolomite", left_D, right_D),
        ("Halite", left_H, right_H),
        ("PIGE", left_P, right_P)
    ]

    for litho, left, right in lithologies:
        if litho in LITHO_COLORS and (right > left).any():
            ax.fill_betweenx(
                depth,
                left,
                right,
                facecolor=LITHO_COLORS[litho],
                edgecolor="black", linewidth=0.3,
                label=litho
            )

    # Configuration du graphique
    ax.set_xlim(0, 1)

# Fixer les bornes de profondeur exactement au min/max
    ax.set_ylim(depth.min(), depth.max())

# Inverser l'axe Y (comme habituellement en géologie)
    ax.invert_yaxis()

# Labels avec couleur jaune
    ax.set_xlabel("Proportion", fontsize=4, color="#FCC13E")
    ax.set_ylabel("Profondeur (m)", fontsize=4, color="#FCC13E")
    ax.set_title("Profil Lithologique", fontsize=9, fontweight="bold", color="#FCC13E")
    ax.set_xlim(0, 1)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis='x', colors="#FCC13E", labelsize=7, top=True, bottom=False)
    ax.set_xlabel("Proportion", fontsize=10, color="#FCC13E")
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='y', colors="#FCC13E",color="#FCC13E", labelsize=7)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.9, 1),
        frameon=True,
        fontsize=6,
        title="Lithologies",
        title_fontsize=8,
        labelcolor="#FCC13E"  
        )
    plt.setp(ax.get_legend().get_title(), color="#FCC13E")
    
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    
    for spine in ax.spines.values():
        spine.set_edgecolor("#0D1B26")
        spine.set_linewidth(1.0)
        
    st.pyplot(fig)

# -------------------- Interface utilisateur --------------------
st.title("🛢️ LithoVision Pro")
st.markdown("---")
from PIL import Image
import streamlit as st


# Charger le logo
logo = Image.open(r"C:\Users\pc\Desktop\nolith_logo.png")

with st.sidebar:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 80%; /* ajuste à 80% de la largeur pour garder qualité */
        }
        [data-testid="stSidebar"] h2 {
            text-align: center;
            font-weight: bold;
            color: #FCC13E;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.image(logo)  # pas de width ici, on laisse CSS gérer
    # st.markdown("<h2>NφLITH</h2>", unsafe_allow_html=True)
    st.markdown("---")




with st.sidebar:
    st.header("Configuration")
    up = st.file_uploader("Importer les données du puits", type=["csv","xlsx"])
    st.header("Paramètres")
    vcl_cutoff = st.slider("VshCut-Off", 0.1, 0.9, 0.4, 0.01, 
                          help="Seuil de coupure pour la fraction d'argile (PIGE=0 si VCL>cutoff)")
    run = st.button("Lancer la prédiction")

st.set_page_config(
    page_title="LithoVision Pro",
    layout="wide",
    initial_sidebar_state="expanded"  # 👈 toujours ouverte au lancement
)


if up:
    df = process_file(up)
    if df is not None:
        st.subheader("Aperçu des données")
        st.dataframe(df.head(10), use_container_width=True)
        
        if 'DEPTH' not in df.columns:
            st.error("La colonne 'DEPTH' est requise dans le fichier.")
        elif run:
            with st.spinner("Prédictions en cours..."):
                models = load_models()
                if models is None:
                    st.stop()
                res = run_predictions(df.copy(), models, VCL_cutoff=vcl_cutoff)
                if res is not None:
                    plot_stacked(res)
                    st.success("Terminé ✅")
else:
    st.info("Veuillez importer un fichier CSV ou Excel.")



st.markdown("---")
st.subheader("⭐ Donnez-nous votre avis")

# Système de notation par étoiles
rating = st.slider("Évaluez l'outil :", 1, 5, 3)

# Champ commentaire
comment = st.text_area("Vos commentaires / suggestions :")

if st.button("Envoyer l'avis"):
    st.success("✅ Merci pour votre retour !")
    # Ici tu peux stocker les données dans un fichier ou une base
    with open("feedback.txt", "a", encoding="utf-8") as f:
        f.write(f"Note: {rating}/5 | Commentaire: {comment}\n")
