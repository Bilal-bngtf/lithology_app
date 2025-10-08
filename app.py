import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# Config de la page avec ton logo comme icône

st.set_page_config(
    page_title="LithoVision Pro",
    page_icon="src/assets/logo.png",  # ton favicon déjà prêt
    layout="wide"
)

# Nouvelle palette adaptée au logo
theme = {
    "primary": "#009688",   # Turquoise (cube)
    "secondary": "#162C3E", # Sidebar bleu nuit
    "background": "#0D1B26",# Fond clair
    "text": "#333333",      # Texte foncé lisible
    "accent": "#FCC13E"     # Jaune doré (symbole central)
}

# ----- Barre horizontale (option 1: supprimer totalement) -----
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
    "VCL": "#54381F",
    "Quartz": "#F0AF79",
    "Igneous": "#73003F",
    "PIGE": "#06633F",
    "Calcite": "#87CEEB",  # Bleu clair
    "Anhydrite": "#5A135A",  # Violet foncé
    "Dolomite": "#205C7E",  # Bleu foncé
    "Halite": "#AAAAFF",  # Bleu pâle
    "Autres_litho": "#D3D3D3"  # Gris clair
}

@st.cache_resource
def load_models():
    """Charge les modèles pré-entraînés"""
    try:
        # base_path = Path(__file__).resolve().parent.parent / "models" / "trained"
        base_path = Path("src/models/trained")
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
    import pandas as pd
    import streamlit as st
    import logging

    logger = logging.getLogger(__name__)

    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("⚠️ Format de fichier non supporté. Utilisez CSV ou XLSX.")
            return None

        # Vérifie si le fichier contient des colonnes
        if df.empty:
            st.error("⚠️ Le fichier est vide.")
            return None

        return df

    except Exception as e:
        msg = f"Erreur de traitement du fichier: {str(e)}"
        logger.error(msg)
        st.error(f"❌ Erreur de lecture du fichier : {e}")
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

        features_autres_reg = ['CALX', 'GR', 'CNC', 'DTCQI', 'K', 'KTH', 'TH', 'ZDEN']
        mask_autres = df["Autres_cls"] == 1
        df["Autres_reg"] = 0.0
        if mask_autres.any():
            df.loc[mask_autres, "Autres_reg"] = models["Autres_reg"].predict(
                df.loc[mask_autres, features_autres_reg]
            )

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

        # ----------------------- PIGE -----------------------
        features_pige = ['CALX', 'GR', 'CNC', 'DTCQI', 'K', 'KTH', 'TH', 'ZDEN', 'VCL_pred']
        df['PIGE_pred'] = models['PIGE_reg'].predict(df[features_pige])

        # Correction règles géologiques
        df["PIGE_final"] = df["PIGE_pred"]
        df.loc[df["VCL_pred"] > VCL_cutoff, "PIGE_final"] = 0

        return df

    except Exception as e:
        logger.error(f"Erreur lors des prédictions: {str(e)}")
        st.error(f"Erreur de prédiction: {str(e)}")
        return None

def plot_curves_and_lithology(results: pd.DataFrame):
    """Affiche les courbes et la colonne lithologique avec la lithologie dominante"""
    df = results.copy()

    # Vérification des colonnes requises
    required_columns = ['DEPTH', 'VCL_pred', 'Quartz_pred', 'PIGE_final']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Colonnes manquantes dans les données : {', '.join(missing_columns)}")
        return

    # Vérifier les colonnes optionnelles
    for col in ['Igneous_reg', 'Autres_reg', 'Final_lithology']:
        if col not in df.columns:
            if col == 'Final_lithology':
                df[col] = np.nan
            else:
                df[col] = 0.0

    # Préparer les données et trier par profondeur
    depth = df['DEPTH'].values
    order = np.argsort(depth)
    depth = depth[order]
    
    VSH = df['VCL_pred'].values[order].clip(0, 1)
    Quartz_raw = df['Quartz_pred'].values[order].clip(0, 1)
    PIGE_original = df['PIGE_final'].values[order].clip(0, 1)
    Igneous = df['Igneous_reg'].values[order].clip(0, 1)
    
    # IMPORTANT: Autres_reg représente la SOMME de toutes les autres lithologies
    Autres_reg = df['Autres_reg'].values[order].clip(0, 1)
    
    # Final_lithology nous dit QUELLE lithologie spécifique c'est (Calcite, Dolomite, etc.)
    Final_lithology = df['Final_lithology'].values[order]
    
    # CONTRAINTE: PIGE = 0 si VCL > cutoff OU Igneous > 0 OU Autres_litho > 0
    vcl_cutoff = 0.40
    PIGE = PIGE_original.copy()
    mask_pige_zero = (VSH > vcl_cutoff) | (Igneous > 0) | (Autres_reg > 0)
    PIGE[mask_pige_zero] = 0.0
    
    Quartz_PIGE = (Quartz_raw + PIGE).clip(0, 1)  # Quartz = Quartz + PIGE

    # Calculer la lithologie dominante pour chaque profondeur
    # Priorité: VCL >= autres (si égalité, VCL gagne)
    litho_matrix = np.column_stack([VSH, Quartz_PIGE, Igneous, Autres_reg])
    litho_names_priority = ['VCL', 'Quartz', 'Igneous', 'Autres']
    
    # Trouver l'indice du maximum
    dominant_litho_idx = np.argmax(litho_matrix, axis=1)
    max_values = np.max(litho_matrix, axis=1)
    
    # Si VCL >= max_value, forcer VCL comme dominant
    vcl_dominant_mask = VSH >= max_values
    dominant_litho_idx[vcl_dominant_mask] = 0  # 0 = VCL
    
    dominant_litho = np.array([litho_names_priority[idx] for idx in dominant_litho_idx])
    
    # Pour les "Autres", déterminer la lithologie spécifique à partir de Final_lithology
    autres_mask = dominant_litho == 'Autres'
    
    # Mapping des valeurs de Final_lithology vers les noms
    int_to_name = {0: "Dolomite", 1: "Anhydrite", 2: "Halite", 3: "Calcite"}
    
    # Compteurs pour diagnostic
    debug_counts = {"Total_Autres": autres_mask.sum(), "NaN": 0, "Mapped": 0, "Unknown": 0}
    lithology_found = {"Calcite": 0, "Dolomite": 0, "Anhydrite": 0, "Halite": 0, "Autres_litho": 0}
    
    if autres_mask.any():
        for i in np.where(autres_mask)[0]:
            final_val = Final_lithology[i]
            
            # Si Final_lithology est défini, on utilise la lithologie spécifique
            if pd.isna(final_val):
                dominant_litho[i] = "Autres_litho"
                debug_counts["NaN"] += 1
                lithology_found["Autres_litho"] += 1
            else:
                # Essayer de convertir en int/float
                try:
                    # Si c'est un nombre (int, float, ou string numérique)
                    if isinstance(final_val, (int, np.integer, float, np.floating)):
                        litho_code = int(final_val)
                        litho_name = int_to_name.get(litho_code, "Autres_litho")
                        debug_counts["Mapped"] += 1
                    elif isinstance(final_val, str):
                        # Essayer de convertir string en int
                        try:
                            litho_code = int(float(final_val))
                            litho_name = int_to_name.get(litho_code, "Autres_litho")
                            debug_counts["Mapped"] += 1
                        except ValueError:
                            # Si c'est un string non-numérique
                            final_lower = final_val.lower()
                            if 'calcite' in final_lower:
                                litho_name = "Calcite"
                            elif 'anhydrite' in final_lower:
                                litho_name = "Anhydrite"
                            elif 'dolomite' in final_lower:
                                litho_name = "Dolomite"
                            elif 'halite' in final_lower:
                                litho_name = "Halite"
                            else:
                                litho_name = "Autres_litho"
                                debug_counts["Unknown"] += 1
                            debug_counts["Mapped"] += 1
                    else:
                        litho_name = "Autres_litho"
                        debug_counts["Unknown"] += 1
                    
                    dominant_litho[i] = litho_name
                    lithology_found[litho_name] = lithology_found.get(litho_name, 0) + 1
                    
                except Exception as e:
                    dominant_litho[i] = "Autres_litho"
                    debug_counts["Unknown"] += 1
                    lithology_found["Autres_litho"] += 1
    
    # Stocker les infos de debug pour affichage plus tard
    debug_info = {
        "debug_counts": debug_counts,
        "lithology_found": lithology_found,
        "sample_values": Final_lithology[autres_mask][:10] if autres_mask.any() else []
    }

    # Limites de profondeur
    depth_min = depth.min()
    depth_max = depth.max()

    # Création de la figure avec gridspec
    fig = plt.figure(figsize=(14, 60))
    fig.patch.set_facecolor("#0D1B26")
    gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 0.05, 1.2], wspace=0.4)

    # ============ TRACK 1: VSH ============
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(VSH, depth, color="#54381F", linewidth=1.5)
    ax1.fill_betweenx(depth, 0, VSH, facecolor="#54381F", alpha=0.9, edgecolor="black", linewidth=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(depth_max, depth_min)
    ax1.set_title("VSH", fontsize=11, fontweight="bold", color="#FCC13E", pad=15)
    ax1.xaxis.set_ticks_position("top")
    ax1.xaxis.set_label_position("top")
    ax1.tick_params(axis='x', labelsize=8, colors="#FFFFFF", top=True, bottom=False)
    ax1.tick_params(axis='y', labelsize=0)
    ax1.grid(True, alpha=0.3, color="#0E4470", linestyle='--', linewidth=0.3, axis='x')
    ax1.set_facecolor("#0D1B26")

    # ============ TRACK 2: Quartz + PIGE ============
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2.fill_betweenx(depth, 0, PIGE, facecolor="#009688", alpha=1, edgecolor="black", 
                      linewidth=0.3, label="PIGE")
    ax2.fill_betweenx(depth, PIGE, Quartz_PIGE, facecolor="#F0AF79", alpha=0.9, 
                      edgecolor="black", linewidth=0.3, label="Quartz")
    ax2.plot(Quartz_PIGE, depth, color="#000000", linewidth=1, linestyle="-", alpha=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(depth_max, depth_min)
    ax2.set_title("PIGE + Quartz", fontsize=11, fontweight="bold", color="#FCC13E", pad=15)
    ax2.xaxis.set_ticks_position("top")
    ax2.xaxis.set_label_position("top")
    ax2.tick_params(axis='x', labelsize=8, colors="#FFFFFF", top=True, bottom=False)
    ax2.tick_params(axis='y', labelsize=0)
    ax2.grid(True, alpha=0.3, color='#0E4470', linestyle='--', linewidth=0.3, axis='x')
    ax2.set_facecolor("#0D1B26")

    # ============ TRACK 3: Igneous ============
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    ax3.fill_betweenx(depth, 0, Igneous, facecolor="#73003F", alpha=0.9, 
                      edgecolor="black", linewidth=0.3)
    ax3.plot(Igneous, depth, color="#000000", linewidth=1, linestyle="-", alpha=0.5)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(depth_max, depth_min)
    ax3.set_title("Igneous", fontsize=11, fontweight="bold", color="#FCC13E", pad=15)
    ax3.xaxis.set_ticks_position("top")
    ax3.xaxis.set_label_position("top")
    ax3.tick_params(axis='x', labelsize=8, colors="#FFFFFF", top=True, bottom=False)
    ax3.tick_params(axis='y', labelsize=0)
    ax3.grid(True, alpha=0.3, color='#0E4470', linestyle='--', linewidth=0.3, axis='x')
    ax3.set_facecolor("#0D1B26")

    # ============ TRACK 4: Autres Lithologies (Autres_reg TOTAL) ============
    ax4 = fig.add_subplot(gs[0, 3], sharey=ax1)
    
    # Afficher simplement Autres_reg en couleur neutre
    ax4.fill_betweenx(depth, 0, Autres_reg, facecolor="#D3D3D3", alpha=0.8, 
                      edgecolor="black", linewidth=0.3, label="Autres Lithologies")
    ax4.plot(Autres_reg, depth, color="#000000", linewidth=1, linestyle="-", alpha=0.5)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(depth_max, depth_min)
    ax4.set_title("Autres Lithologies", fontsize=11, fontweight="bold", color="#FCC13E", pad=15)
    ax4.xaxis.set_ticks_position("top")
    ax4.xaxis.set_label_position("top")
    ax4.tick_params(axis='x', labelsize=8, colors="#FFFFFF", top=True, bottom=False)
    ax4.tick_params(axis='y', labelsize=0)
    ax4.grid(True, alpha=0.3, color='#0E4470', linestyle='--', linewidth=0.3, axis='x')
    ax4.set_facecolor("#0D1B26")

    # ============ TRACK 5: Profondeur ============
    ax5 = fig.add_subplot(gs[0, 4], sharey=ax1)
    ax5.set_xlim(0, 1)
    ax5.set_ylim(depth_max, depth_min)
    ax5.set_title("Depth (m)", fontsize=11, fontweight="bold", color="#FCC13E", pad=30)
    ax5.yaxis.set_major_locator(MultipleLocator(10))
    ax5.yaxis.set_minor_locator(MultipleLocator(2))
    ax5.tick_params(axis='y', labelsize=8, colors="#FFFFFF", which='both', left=False, right=True, labelleft=False, labelright=True)
    ax5.tick_params(axis='x', labelsize=0, top=False, bottom=False)
    ax5.grid(True, alpha=0.4, color='#0E4470', linestyle='-', linewidth=0.5, axis='y', which='major')
    ax5.grid(True, alpha=0.2, color='#0E4470', linestyle='--', linewidth=0.3, axis='y', which='minor')
    ax5.set_xticks([])
    ax5.set_facecolor("#FCC13E")
    ax5.yaxis.set_label_position("right")

    # ============ TRACK 6: Colonne Lithologique DOMINANTE ============
    ax6 = fig.add_subplot(gs[0, 5], sharey=ax1)
    
    # Debug: Compter les lithologies uniques dans dominant_litho
    unique_lithos = np.unique(dominant_litho)
    litho_color_check = {}
    for litho in unique_lithos:
        color = LITHO_COLORS.get(litho, '#D3D3D3')
        litho_color_check[litho] = color
    
    # Grouper les segments continus de même lithologie
    current_litho = dominant_litho[0]
    start_depth = depth[0]
    
    for i in range(1, len(depth)):
        if dominant_litho[i] != current_litho or i == len(depth) - 1:
            # Fin du segment
            end_depth = depth[i] if i == len(depth) - 1 else depth[i-1]
            color = LITHO_COLORS.get(current_litho, '#D3D3D3')
            ax6.fill_between([0, 1], start_depth, end_depth, 
                             facecolor=color, edgecolor='black', linewidth=0.5)
            # Nouveau segment
            current_litho = dominant_litho[i]
            start_depth = depth[i]
    
    # Dernier segment
    color = LITHO_COLORS.get(current_litho, '#D3D3D3')
    ax6.fill_between([0, 1], start_depth, depth[-1], 
                     facecolor=color, edgecolor='black', linewidth=0.5)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(depth_max, depth_min)
    ax6.set_title("Lithologie\nDominante", fontsize=11, fontweight="bold", color="#FCC13E", pad=20)
    ax6.set_xticks([])
    ax6.tick_params(axis='y', labelsize=0)
    ax6.set_facecolor("#0D1B26")
    
    # Ajouter la légende
    from matplotlib.patches import Patch
    litho_legend_names = ['VCL', 'Quartz', 'Igneous', 'Calcite', 'Anhydrite', 'Dolomite', 'Halite', 'Autres_litho']
    legend_elements = [Patch(facecolor=LITHO_COLORS[name], edgecolor='black', label=name) 
                      for name in litho_legend_names if name in LITHO_COLORS]
    ax6.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.7, 1.0),
               fontsize=7, frameon=True, facecolor='#0D1B26', edgecolor="#FFFFFF", 
               labelcolor="#FFFFFF", title='Lithologies', title_fontsize=8)
    plt.setp(ax6.get_legend().get_title(), color="#FCC13E")
    
    # Stocker les infos de couleurs pour le diagnostic
    debug_info['litho_color_check'] = litho_color_check

    # Style général - bordures
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        for spine in ax.spines.values():
            spine.set_edgecolor("#FFFFFF")
            spine.set_linewidth(1.2)

    plt.tight_layout()
    st.pyplot(fig)

    # ============ Statistiques ============
    st.markdown("### 📊 Statistiques des prédictions")
    
    # DIAGNOSTIC: Afficher les infos de mapping des autres lithologies
    st.markdown("#### 🔍 Diagnostic - Mapping des Autres Lithologies")
    st.write(f"**Total de points où 'Autres' domine:** {debug_info['debug_counts']['Total_Autres']}")
    st.write(f"**Valeurs NaN (non classifiées):** {debug_info['debug_counts']['NaN']}")
    st.write(f"**Valeurs mappées avec succès:** {debug_info['debug_counts']['Mapped']}")
    st.write(f"**Valeurs inconnues:** {debug_info['debug_counts']['Unknown']}")
    
    st.write("**Lithologies spécifiques trouvées:**")
    for litho, count in debug_info['lithology_found'].items():
        if count > 0:
            st.write(f"- {litho}: {count} points")
    
    if len(debug_info['sample_values']) > 0:
        st.write(f"**Échantillon de Final_lithology (10 premiers):** {list(debug_info['sample_values'])}")
    
    # Afficher le mapping couleur/lithologie
    st.write("**Couleurs assignées dans Track 6:**")
    for litho, color in debug_info['litho_color_check'].items():
        st.markdown(f"- {litho}: <span style='color:{color}; font-weight:bold;'>███</span> {color}", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comptage des lithologies dominantes
    from collections import Counter
    litho_counts = Counter(dominant_litho)
    
    st.write("**Répartition des lithologies dominantes:**")
    for litho, count in litho_counts.most_common():
        percentage = (count / len(dominant_litho)) * 100
        color = LITHO_COLORS.get(litho, '#D3D3D3')
        st.markdown(f"- <span style='color:{color}; font-weight:bold;'>███</span> {litho}: {count} points ({percentage:.1f}%)", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("VCL moyen", f"{VSH.mean():.2%}")
        st.metric("VCL max", f"{VSH.max():.2%}")
    
    with col2:
        st.metric("Quartz+PIGE moyen", f"{Quartz_PIGE.mean():.2%}")
        st.metric("PIGE moyen", f"{PIGE.mean():.2%}")
    
    with col3:
        st.metric("Igneous moyen", f"{Igneous.mean():.2%}")
        st.metric("Igneous max", f"{Igneous.max():.2%}")
    
    with col4:
        st.metric("Autres lithologies", f"{Autres_reg.mean():.2%}")
        st.metric("Autres max", f"{Autres_reg.max():.2%}")

# -------------------- Interface utilisateur --------------------
st.title("LithoVision Pro")
st.markdown("---")

with st.sidebar:
    st.header("Configuration")
    up = st.file_uploader("Importer les données du puits", type=["csv","xlsx"])
    st.header("Paramètres")
    vcl_cutoff = st.slider("VshCut-Off", 0.1, 0.9, 0.4, 0.01, 
                          help="Seuil de coupure pour la fraction d'argile (PIGE=0 si VCL>cutoff)")
    run = st.button("Lancer la prédiction")

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
                    plot_curves_and_lithology(res)
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
    with open("feedback.txt", "a", encoding="utf-8") as f:
        f.write(f"Note: {rating}/5 | Commentaire: {comment}\n")

import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

creds_dict = st.secrets["google"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)

sheet = client.open("Lithology Feedback").sheet1
sheet.append_row([rating, comment])
