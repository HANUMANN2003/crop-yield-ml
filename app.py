import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Yield ML Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8faf5; }
    .stMetric { background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
    .block-container { padding-top: 2rem; }
    h1 { color: #2d6a4f; font-weight: 800; }
    h2, h3 { color: #40916c; }
    .st-emotion-cache-1v0mbdj { border-radius: 12px; }
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e0f0e9;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .prediction-box {
        background: linear-gradient(135deg, #2d6a4f, #40916c);
        color: white;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 16px 0;
        box-shadow: 0 4px 20px rgba(45,106,79,0.35);
    }
    .high-yield { background: linear-gradient(135deg, #1b4332, #2d6a4f); }
    .low-yield  { background: linear-gradient(135deg, #774936, #b5541f); }
    .sidebar-info {
        background: #e8f5e9;
        border-left: 4px solid #40916c;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.85rem;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    reg = joblib.load("regression_model.pkl")
    clf = joblib.load("classification_model.pkl")
    le  = joblib.load("label_encoders.pkl")
    return reg, clf, le

@st.cache_data
def load_data():
    return pd.read_csv("crop_yield_cleaned.csv")

reg_model, clf_model, encoders = load_models()
df = load_data()

FEATURES = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest',
            'Fertilizer_Used', 'Irrigation_Used', 'Rain_per_Day',
            'Climate_Index', 'Region_enc', 'Soil_Type_enc',
            'Crop_enc', 'Weather_Condition_enc']

YIELD_MEDIAN = 4.651808

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/wheat.png", width=72)
    st.title("🌾 Crop Yield AI")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Overview", "🔮 Predict Yield", "📊 EDA Dashboard", "📈 Model Insights"])
    st.markdown("---")
    st.markdown("""
    <div class='sidebar-info'>
    <b>Dataset</b><br>
    🗂 1,000,000 rows<br>
    📌 10 features<br>
    🌍 4 Regions · 6 Crops<br>
    🤖 HistGradientBoosting<br><br>
    <b>Model Performance</b><br>
    📉 Regression R² = 0.88<br>
    🎯 Classification Acc = 90%
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🌾 Crop Yield Prediction Dashboard")
    st.markdown("##### End-to-end Machine Learning Pipeline | Agriculture Domain")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Records",   "1,000,000")
    c2.metric("🌱 Crops Covered",   "6")
    c3.metric("🌍 Regions",         "4")
    c4.metric("📐 Features Used",   "11")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Project Pipeline")
        steps = [
            ("1️⃣", "Data Collection",       "Kaggle Agriculture Dataset"),
            ("2️⃣", "Data Cleaning",          "Nulls · Duplicates · Outliers · Typos"),
            ("3️⃣", "EDA",                    "Distributions · Correlations · Group Analysis"),
            ("4️⃣", "Feature Engineering",    "Rain/Day · Climate Index · Label Encoding"),
            ("5️⃣", "Model Training",         "HistGradientBoosting (Reg + Clf)"),
            ("6️⃣", "Deployment",             "Streamlit Interactive Dashboard"),
        ]
        for icon, title, desc in steps:
            st.markdown(f"**{icon} {title}** — _{desc}_")

    with col2:
        st.subheader("🏆 Model Results")
        results = pd.DataFrame({
            "Task":       ["Regression", "Classification"],
            "Model":      ["HistGBR", "HistGBC"],
            "Key Metric": ["R² = 0.880", "Accuracy = 90%"],
            "MAE / F1":   ["0.442 tons/ha", "F1 = 0.90"],
            "Data Size":  ["1M rows", "1M rows"],
        })
        st.dataframe(results, use_container_width=True, hide_index=True)

        st.subheader("📊 Yield Distribution Preview")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(df['Yield_tons_per_hectare'], bins=60, color='#40916c', edgecolor='white', alpha=0.85)
        ax.axvline(YIELD_MEDIAN, color='#d62828', linestyle='--', linewidth=1.8, label=f'Median = {YIELD_MEDIAN:.2f}')
        ax.set_xlabel('Yield (tons/ha)'); ax.set_ylabel('Count')
        ax.set_title('Target Variable Distribution'); ax.legend()
        fig.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Yield":
    st.title("🔮 Predict Crop Yield")
    st.markdown("Fill in the farm conditions below and get an instant AI prediction.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌍 Location & Crop")
        region   = st.selectbox("Region",         list(encoders['Region'].classes_))
        soil     = st.selectbox("Soil Type",       list(encoders['Soil_Type'].classes_))
        crop     = st.selectbox("Crop",            list(encoders['Crop'].classes_))
        weather  = st.selectbox("Weather Condition", list(encoders['Weather_Condition'].classes_))

    with col2:
        st.subheader("🌧️ Climate Conditions")
        rainfall = st.slider("Rainfall (mm)",          100, 1000, 550, step=10)
        temp     = st.slider("Temperature (°C)",        15,   40,  27, step=1)
        days     = st.slider("Days to Harvest",         60,  149,  105, step=1)

    with col3:
        st.subheader("🌿 Farm Inputs")
        fertilizer = st.radio("Fertilizer Used?", ["Yes", "No"], horizontal=True)
        irrigation = st.radio("Irrigation Used?", ["Yes", "No"], horizontal=True)

        st.markdown("---")
        st.subheader("📐 Derived Features")
        rain_per_day   = rainfall / days
        climate_index  = (temp * rainfall) / 1000
        st.info(f"**Rain per Day:** {rain_per_day:.2f} mm/day")
        st.info(f"**Climate Index:** {climate_index:.2f}")

    st.markdown("---")
    if st.button("🚀 Predict Now", use_container_width=True, type="primary"):
        fert_int = 1 if fertilizer == "Yes" else 0
        irri_int = 1 if irrigation == "Yes" else 0

        input_data = pd.DataFrame([{
            'Rainfall_mm':          rainfall,
            'Temperature_Celsius':  temp,
            'Days_to_Harvest':      days,
            'Fertilizer_Used':      fert_int,
            'Irrigation_Used':      irri_int,
            'Rain_per_Day':         rain_per_day,
            'Climate_Index':        climate_index,
            'Region_enc':           encoders['Region'].transform([region])[0],
            'Soil_Type_enc':        encoders['Soil_Type'].transform([soil])[0],
            'Crop_enc':             encoders['Crop'].transform([crop])[0],
            'Weather_Condition_enc':encoders['Weather_Condition'].transform([weather])[0],
        }])

        reg_pred = reg_model.predict(input_data)[0]
        clf_pred = clf_model.predict(input_data)[0]
        clf_prob = clf_model.predict_proba(input_data)[0]

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class='prediction-box'>
                🌾 Predicted Yield<br>
                <span style='font-size:2.6rem'>{reg_pred:.3f}</span><br>
                <span style='font-size:1rem;opacity:0.85'>tons per hectare</span>
            </div>""", unsafe_allow_html=True)

        with r2:
            label     = "High Yield 🟢" if clf_pred == 1 else "Low Yield 🔴"
            box_class = "high-yield" if clf_pred == 1 else "low-yield"
            conf      = clf_prob[clf_pred] * 100
            st.markdown(f"""
            <div class='prediction-box {box_class}'>
                🎯 Classification<br>
                <span style='font-size:2.2rem'>{label}</span><br>
                <span style='font-size:1rem;opacity:0.85'>Confidence: {conf:.1f}%</span>
            </div>""", unsafe_allow_html=True)

        # Confidence bar
        st.markdown("#### 📊 Class Probability")
        prob_df = pd.DataFrame({'Class': ['Low Yield', 'High Yield'], 'Probability': clf_prob})
        fig, ax = plt.subplots(figsize=(6, 2))
        colors = ['#b5541f', '#2d6a4f']
        ax.barh(prob_df['Class'], prob_df['Probability'], color=colors, height=0.5)
        ax.set_xlim(0, 1); ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
        for i, v in enumerate(prob_df['Probability']):
            ax.text(v + 0.02, i, f'{v:.2%}', va='center', fontweight='bold')
        ax.set_xlabel('Probability'); fig.tight_layout()
        st.pyplot(fig); plt.close()

        # Comparison to dataset average
        st.markdown("#### 📌 How does this compare?")
        avg = df.groupby('Crop')['Yield_tons_per_hectare'].mean()[crop]
        delta = reg_pred - avg
        st.metric(f"Avg yield for {crop}", f"{avg:.3f} tons/ha",
                  delta=f"{delta:+.3f} vs your prediction")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📦 Distributions", "🌱 Crop Analysis", "🌍 Region Analysis", "🔗 Correlations"])

    with tab1:
        st.subheader("Numerical Feature Distributions")
        num_cols = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest', 'Yield_tons_per_hectare']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        colors = ['#40916c', '#52b788', '#74c69d', '#95d5b2']
        for ax, col, color in zip(axes.flatten(), num_cols, colors):
            ax.hist(df[col], bins=50, color=color, edgecolor='white', alpha=0.9)
            ax.set_title(col, fontweight='bold'); ax.set_ylabel('Count')
        fig.suptitle("Feature Distributions", fontsize=14, fontweight='bold')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        st.subheader("Yield Analysis by Crop")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            crop_avg = df.groupby('Crop')['Yield_tons_per_hectare'].mean().sort_values()
            bars = ax.barh(crop_avg.index, crop_avg.values, color='#40916c', edgecolor='white')
            ax.bar_label(bars, fmt='%.2f', padding=4)
            ax.set_xlabel('Avg Yield (tons/ha)'); ax.set_title('Avg Yield by Crop', fontweight='bold')
            fig.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby('Crop')['High_Yield'].mean().sort_values().plot(
                kind='barh', ax=ax, color='#52b788', edgecolor='white')
            ax.set_xlabel('High Yield Rate'); ax.set_title('High Yield Rate by Crop', fontweight='bold')
            ax.axvline(0.5, color='red', linestyle='--', linewidth=1.2)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("Yield Distribution per Crop (Boxplot)")
        fig, ax = plt.subplots(figsize=(12, 5))
        order = df.groupby('Crop')['Yield_tons_per_hectare'].median().sort_values().index
        sns.boxplot(data=df, x='Crop', y='Yield_tons_per_hectare', order=order,
                    palette='Greens', ax=ax)
        ax.set_title('Yield Distribution by Crop', fontweight='bold')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        st.subheader("Yield by Region & Weather")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=df, x='Region', y='Yield_tons_per_hectare',
                    hue='Weather_Condition', palette=['#52b788','#40916c','#1b4332'], ax=ax)
        ax.set_title('Yield by Region & Weather Condition', fontweight='bold')
        ax.legend(title='Weather')
        fig.tight_layout(); st.pyplot(fig); plt.close()

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4))
            df.groupby('Region')['Yield_tons_per_hectare'].mean().plot(
                kind='bar', ax=ax, color='#40916c', rot=0, edgecolor='white')
            ax.set_title('Avg Yield by Region', fontweight='bold')
            ax.set_ylabel('Avg Yield (tons/ha)')
            fig.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(5, 4))
            fert = df.groupby('Fertilizer_Used')['Yield_tons_per_hectare'].mean()
            irri = df.groupby('Irrigation_Used')['Yield_tons_per_hectare'].mean()
            x = np.arange(2)
            ax.bar(x - 0.2, fert.values, 0.35, label='Fertilizer', color='#40916c')
            ax.bar(x + 0.2, irri.values, 0.35, label='Irrigation',  color='#74c69d')
            ax.set_xticks(x); ax.set_xticklabels(['No (0)', 'Yes (1)'])
            ax.set_ylabel('Avg Yield'); ax.set_title('Effect of Fertilizer & Irrigation', fontweight='bold')
            ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        st.subheader("Correlation Heatmap")
        num_df = df[['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest',
                     'Fertilizer_Used', 'Irrigation_Used',
                     'Rain_per_Day', 'Climate_Index', 'Yield_tons_per_hectare']]
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='Greens',
                    linewidths=0.5, ax=ax, annot_kws={'size': 9})
        ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=13)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("Rainfall vs Yield (Scatter)")
        sample = df.sample(5000, random_state=42)
        fig, ax = plt.subplots(figsize=(10, 4))
        scatter = ax.scatter(sample['Rainfall_mm'], sample['Yield_tons_per_hectare'],
                             c=sample['Temperature_Celsius'], cmap='YlGn', alpha=0.5, s=12)
        plt.colorbar(scatter, ax=ax, label='Temperature (°C)')
        ax.set_xlabel('Rainfall (mm)'); ax.set_ylabel('Yield (tons/ha)')
        ax.set_title('Rainfall vs Yield (colored by Temperature)', fontweight='bold')
        fig.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.title("📈 Model Insights")
    st.markdown("---")

    tab1, tab2 = st.tabs(["🌲 Feature Importance", "🎯 Model Performance"])

    with tab1:
        st.subheader("🌲 Feature Importance")

        st.info("""
        This project uses HistGradientBoosting models.

        HistGradientBoostingRegressor and HistGradientBoostingClassifier
        do not provide the feature_importances_ attribute.

        Model performance metrics are available in the next tab.
        """)

    with tab2:
        st.subheader("Model Performance Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Regression R²", "0.880", "↑ Great fit")
        col2.metric("Regression MAE", "0.442", "tons/ha error")
        col3.metric("Clf Accuracy", "90%", "↑ Balanced classes")
        col4.metric("Clf F1 Score", "0.90", "Both classes")

        st.markdown("---")

        st.subheader("Why HistGradientBoosting?")

        compare = pd.DataFrame({
            "Model": ["RandomForest", "HistGradientBoosting"],
            "R² Score": [0.872, 0.880],
            "MAE": [0.456, 0.442],
            "Training Speed": ["~3 min (100k)", "~2 min (1M)"],
            "Memory": ["High", "Low"],
            "Large Data": ["❌ Slow", "✅ Fast"],
        })

        st.dataframe(compare, use_container_width=True, hide_index=True)

        st.markdown("---")

        st.subheader("Confusion Matrix Interpretation")

        st.markdown("""
| | Predicted Low | Predicted High |
|---|---|---|
| **Actual Low** | ✅ 90k True Negative | ❌ 10k False Positive |
| **Actual High** | ❌ 10k False Negative | ✅ 90k True Positive |

- **90% of predictions are correct**
- **Classes are balanced**
- **Model generalises well**
""")
