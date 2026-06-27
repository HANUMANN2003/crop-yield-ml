import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

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

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

h1,h2,h3 {
    color:#1b4332;
}

div[data-testid="metric-container"] {
    background:white;
    border-radius:18px;
    padding:15px;
    border:none;
    box-shadow:0px 3px 12px rgba(0,0,0,0.08);
}

.stButton button {
    width:100%;
    height:50px;
    border-radius:12px;
    font-weight:700;
}

.hero {
    background:linear-gradient(
        135deg,
        #2d6a4f,
        #52b788
    );
    padding:30px;
    border-radius:20px;
    color:white;
    text-align:center;
    margin-bottom:20px;
}

.prediction-card {
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 3px 12px rgba(0,0,0,0.08);
    text-align:center;
}

.region-card {
    border-radius:14px;
    padding:16px 20px;
    margin-bottom:12px;
    color:white;
    font-family:sans-serif;
}

.rec-card {
    border-radius:12px;
    padding:18px;
    text-align:center;
    margin-bottom:8px;
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

REGION_COLORS = {
    "North": "#2E8B57",
    "South": "#FF7F50",
    "East":  "#4682B4",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2909/2909763.png",
        width=80
    )

    st.title("🌾 Crop Yield AI")

    st.markdown("---")

    page = st.selectbox(
        "📌 Navigation",
        [
            "🏠 Overview",
            "🔮 Predict Yield",
            "📊 EDA Dashboard",
            "📈 Model Insights",
            "🗺️ Region Analysis",       # ← NEW PAGE
        ]
    )

    st.markdown("---")

    st.metric("🎯 Accuracy", "90%")
    st.metric("📈 R² Score", "0.88")

    st.markdown("---")

    st.success("""
    🌱 AI-Powered Agriculture

    • 1M Records

    • 6 Crops

    • 4 Regions

    • HistGradientBoosting
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero">
    <h1>🌾 Smart Crop Yield Analytics</h1>
    <p>
    AI-powered agricultural intelligence platform
    for crop yield prediction and farm optimization
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.title("🌾 Crop Yield Prediction Dashboard")
    st.markdown("##### End-to-end Machine Learning Pipeline | Agriculture Domain")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📦 Records", "1M+")
    c2.metric("🌱 Crops", "6")
    c3.metric("🌍 Regions", "4")
    c4.metric("🎯 Accuracy", "90%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌟 Key Insights")

        st.info("🌧 Rainfall has a strong influence on crop yield.")
        st.info("💧 Irrigation improves productivity significantly.")
        st.info("🌱 Rice and Wheat achieve consistently high yields.")
        st.info("🌍 Climate conditions directly affect harvest outcomes.")

    with col2:
        st.subheader("🏆 Model Results")

        results = pd.DataFrame({
            "Task": ["Regression", "Classification"],
            "Model": ["HistGBR", "HistGBC"],
            "Key Metric": ["R² = 0.880", "Accuracy = 90%"],
            "MAE / F1": ["0.442 tons/ha", "F1 = 0.90"],
            "Data Size": ["1M rows", "1M rows"],
        })

        st.dataframe(results, use_container_width=True, hide_index=True)

        st.subheader("📊 Yield Distribution Preview")

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(
            df['Yield_tons_per_hectare'],
            bins=60,
            color='#40916c',
            edgecolor='white',
            alpha=0.85
        )

        ax.axvline(
            YIELD_MEDIAN,
            color='#d62828',
            linestyle='--',
            linewidth=1.8,
            label=f'Median = {YIELD_MEDIAN:.2f}'
        )

        ax.set_xlabel('Yield (tons/ha)')
        ax.set_ylabel('Count')
        ax.set_title('Target Variable Distribution')
        ax.legend()

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()


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
        st.markdown("---")


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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REGION ANALYSIS  (NEW)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Region Analysis":
    st.title("🗺️ Region-wise Yield Analysis")
    st.markdown("Explore crop yield patterns across **North, South, and East** regions "
                "with interactive charts and smart farming recommendations.")
    st.markdown("---")

    # ── Sidebar filters (scoped to this page) ────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔍 Region Filters")

        regions  = ["All"] + sorted(df["Region"].dropna().unique().tolist())
        sel_region = st.selectbox("Region", regions, key="ra_region")

        crops = ["All"] + sorted(df["Crop"].dropna().unique().tolist())
        sel_crop = st.selectbox("Crop", crops, key="ra_crop")

        soils = ["All"] + sorted(df["Soil_Type"].dropna().unique().tolist())
        sel_soil = st.selectbox("Soil Type", soils, key="ra_soil")

        weathers = ["All"] + sorted(df["Weather_Condition"].dropna().unique().tolist())
        sel_weather = st.selectbox("Weather Condition", weathers, key="ra_weather")

        sel_fertilizer = st.selectbox("Fertilizer Used", ["All", "Yes", "No"], key="ra_fert")
        sel_irrigation  = st.selectbox("Irrigation Used",  ["All", "Yes", "No"], key="ra_irri")

    # ── Apply filters ─────────────────────────────────────────────────────────
    fdf = df.copy()
    if sel_region    != "All": fdf = fdf[fdf["Region"]            == sel_region]
    if sel_crop      != "All": fdf = fdf[fdf["Crop"]              == sel_crop]
    if sel_soil      != "All": fdf = fdf[fdf["Soil_Type"]         == sel_soil]
    if sel_weather   != "All": fdf = fdf[fdf["Weather_Condition"] == sel_weather]
    if sel_fertilizer != "All":
        fdf = fdf[fdf["Fertilizer_Used"] == (sel_fertilizer == "Yes")]
    if sel_irrigation != "All":
        fdf = fdf[fdf["Irrigation_Used"] == (sel_irrigation == "Yes")]

    if fdf.empty:
        st.warning("⚠️ No records match the selected filters. Please adjust the sidebar filters.")
        st.stop()

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.subheader("📊 Summary Statistics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records",       f"{len(fdf):,}")
    k2.metric("Avg Yield (t/ha)",    f"{fdf['Yield_tons_per_hectare'].mean():.2f}")
    k3.metric("Max Yield (t/ha)",    f"{fdf['Yield_tons_per_hectare'].max():.2f}")
    k4.metric("Avg Days to Harvest", f"{fdf['Days_to_Harvest'].mean():.0f}")

    st.markdown("---")

    # ── ROW 1 : Region Cards + Avg Yield Bar ─────────────────────────────────
    col_cards, col_bar = st.columns([1, 1.6])

    with col_cards:
        st.subheader("🗾 Region Overview")
        region_stats = (
            df.groupby("Region")["Yield_tons_per_hectare"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "Avg Yield", "count": "Records"})
            .reset_index()
        )
        for _, row in region_stats.iterrows():
            region_name = row["Region"]
            color  = REGION_COLORS.get(region_name, "#888")
            border = "border: 3px solid #FFD700;" if (sel_region == region_name or sel_region == "All") else "opacity:0.45;"
            st.markdown(
                f"""<div style="background:{color};{border}border-radius:14px;
                            padding:14px 18px;margin-bottom:10px;color:white;">
                  <b style="font-size:18px;">{region_name} Region</b><br>
                  🌾 Avg Yield : <b>{row['Avg Yield']:.2f} t/ha</b><br>
                  📋 Records   : <b>{int(row['Records']):,}</b>
                </div>""",
                unsafe_allow_html=True,
            )

    with col_bar:
        st.subheader("📈 Avg Yield by Region")
        agg = (
            fdf.groupby("Region")["Yield_tons_per_hectare"]
            .mean().reset_index()
            .sort_values("Yield_tons_per_hectare", ascending=False)
        )
        fig = px.bar(
            agg, x="Region", y="Yield_tons_per_hectare",
            color="Region", color_discrete_map=REGION_COLORS,
            text_auto=".2f",
            labels={"Yield_tons_per_hectare": "Avg Yield (t/ha)"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=320,
                          plot_bgcolor="rgba(0,0,0,0)",
                          yaxis=dict(gridcolor="#e0e0e0"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── ROW 2 : Crop-wise + Soil-wise ────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🌽 Crop-wise Avg Yield")
        crop_agg = (
            fdf.groupby(["Crop", "Region"])["Yield_tons_per_hectare"]
            .mean().reset_index()
        )
        fig2 = px.bar(
            crop_agg, x="Crop", y="Yield_tons_per_hectare",
            color="Region", barmode="group",
            color_discrete_map=REGION_COLORS,
            labels={"Yield_tons_per_hectare": "Avg Yield (t/ha)"},
        )
        fig2.update_layout(height=360, plot_bgcolor="rgba(0,0,0,0)",
                           xaxis_tickangle=-30, yaxis=dict(gridcolor="#e0e0e0"))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader("🪨 Soil-wise Avg Yield")
        soil_agg = (
            fdf.groupby(["Soil_Type", "Region"])["Yield_tons_per_hectare"]
            .mean().reset_index()
        )
        fig3 = px.bar(
            soil_agg, x="Soil_Type", y="Yield_tons_per_hectare",
            color="Region", barmode="group",
            color_discrete_map=REGION_COLORS,
            labels={"Yield_tons_per_hectare": "Avg Yield (t/ha)"},
        )
        fig3.update_layout(height=360, plot_bgcolor="rgba(0,0,0,0)",
                           xaxis_tickangle=-30, yaxis=dict(gridcolor="#e0e0e0"))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── ROW 3 : Weather + Fertilizer/Irrigation Impact ───────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("🌤️ Weather Condition vs Yield")
        weather_agg = (
            fdf.groupby("Weather_Condition")["Yield_tons_per_hectare"]
            .mean().reset_index()
            .sort_values("Yield_tons_per_hectare", ascending=True)
        )
        fig4 = px.bar(
            weather_agg, x="Yield_tons_per_hectare", y="Weather_Condition",
            orientation="h", color="Yield_tons_per_hectare",
            color_continuous_scale="Greens", text_auto=".2f",
            labels={"Yield_tons_per_hectare": "Avg Yield (t/ha)",
                    "Weather_Condition": "Weather"},
        )
        fig4.update_layout(height=320, coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

    with c4:
        st.subheader("💧 Fertilizer & Irrigation Impact")
        fdf = fdf.copy()
        fdf["Practice"] = fdf.apply(lambda r:
            "Fert✅ Irr✅" if r["Fertilizer_Used"] and r["Irrigation_Used"] else
            "Fert✅ Irr❌" if r["Fertilizer_Used"] else
            "Fert❌ Irr✅" if r["Irrigation_Used"] else
            "Fert❌ Irr❌", axis=1)
        practice_agg = (
            fdf.groupby("Practice")["Yield_tons_per_hectare"]
            .mean().reset_index()
            .sort_values("Yield_tons_per_hectare", ascending=False)
        )
        fig5 = px.bar(
            practice_agg, x="Practice", y="Yield_tons_per_hectare",
            color="Practice", text_auto=".2f",
            labels={"Yield_tons_per_hectare": "Avg Yield (t/ha)"},
        )
        fig5.update_layout(height=320, showlegend=False,
                           plot_bgcolor="rgba(0,0,0,0)",
                           yaxis=dict(gridcolor="#e0e0e0"))
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # ── ROW 4 : Scatter plots ─────────────────────────────────────────────────
    st.subheader("🌧️ Climate vs Yield")
    sc1, sc2 = st.columns(2)
    sample_fdf = fdf.sample(min(5000, len(fdf)), random_state=42)

    with sc1:
        fig6 = px.scatter(
            sample_fdf, x="Rainfall_mm", y="Yield_tons_per_hectare",
            color="Region", color_discrete_map=REGION_COLORS,
            opacity=0.5, trendline="ols",
            labels={"Rainfall_mm": "Rainfall (mm)",
                    "Yield_tons_per_hectare": "Yield (t/ha)"},
            title="Rainfall vs Yield",
        )
        fig6.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig6, use_container_width=True)

    with sc2:
        fig7 = px.scatter(
            sample_fdf, x="Temperature_Celsius", y="Yield_tons_per_hectare",
            color="Region", color_discrete_map=REGION_COLORS,
            opacity=0.5, trendline="ols",
            labels={"Temperature_Celsius": "Temperature (°C)",
                    "Yield_tons_per_hectare": "Yield (t/ha)"},
            title="Temperature vs Yield",
        )
        fig7.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig7, use_container_width=True)

    st.markdown("---")

    # ── ROW 5 : Days to Harvest by Region & Crop ─────────────────────────────
    st.subheader("🗓️ Days to Harvest by Region & Crop")
    harvest_agg = (
        fdf.groupby(["Crop", "Region"])["Days_to_Harvest"]
        .mean().reset_index()
    )
    fig8 = px.bar(
        harvest_agg, x="Crop", y="Days_to_Harvest",
        color="Region", barmode="group",
        color_discrete_map=REGION_COLORS,
        labels={"Days_to_Harvest": "Avg Days to Harvest"},
    )
    fig8.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)",
                       xaxis_tickangle=-20, yaxis=dict(gridcolor="#e0e0e0"))
    st.plotly_chart(fig8, use_container_width=True)

    st.markdown("---")

    # ── ROW 6 : SMART FARMING RECOMMENDATION ─────────────────────────────────
    st.subheader("🌱 Smart Farming Recommendation")
    st.markdown("Enter your field details to get a **fertilizer & irrigation recommendation** "
                "based on historical yield patterns from similar conditions.")

    r1, r2, r3 = st.columns(3)
    inp_region  = r1.selectbox("Your Region",    sorted(df["Region"].dropna().unique()),       key="rec_region")
    inp_crop    = r2.selectbox("Your Crop",       sorted(df["Crop"].dropna().unique()),         key="rec_crop")
    inp_soil    = r3.selectbox("Your Soil Type",  sorted(df["Soil_Type"].dropna().unique()),    key="rec_soil")

    r4, r5 = st.columns(2)
    inp_rainfall = r4.slider(
        "Expected Rainfall (mm)",
        int(df["Rainfall_mm"].min()), int(df["Rainfall_mm"].max()),
        int(df["Rainfall_mm"].median()), key="rec_rain"
    )
    inp_temp = r5.slider(
        "Expected Temperature (°C)",
        int(df["Temperature_Celsius"].min()), int(df["Temperature_Celsius"].max()),
        int(df["Temperature_Celsius"].median()), key="rec_temp"
    )

    inp_weather = st.selectbox(
        "Expected Weather Condition",
        sorted(df["Weather_Condition"].dropna().unique()),
        key="rec_weather"
    )

    if st.button("🔍 Get Recommendation", type="primary", key="rec_btn"):

        # Filter similar records
        mask = (
            (df["Region"]            == inp_region) &
            (df["Crop"]              == inp_crop) &
            (df["Soil_Type"]         == inp_soil) &
            (df["Weather_Condition"] == inp_weather)
        )
        sub = df[mask]
        if len(sub) < 10:
            sub = df[(df["Region"] == inp_region) & (df["Crop"] == inp_crop)]

        if sub.empty:
            st.warning("⚠️ Not enough similar records to generate a recommendation.")
        else:
            combo = (
                sub.groupby(["Fertilizer_Used", "Irrigation_Used"])
                ["Yield_tons_per_hectare"]
                .mean().reset_index()
                .sort_values("Yield_tons_per_hectare", ascending=False)
                .iloc[0]
            )
            rec_fert = bool(combo["Fertilizer_Used"])
            rec_irri = bool(combo["Irrigation_Used"])
            exp_yield = combo["Yield_tons_per_hectare"]

            low_rain  = inp_rainfall < df["Rainfall_mm"].quantile(0.33)
            high_rain = inp_rainfall > df["Rainfall_mm"].quantile(0.67)

            ra, rb = st.columns(2)

            with ra:
                fert_text  = "✅ Recommended"   if rec_fert else "❌ Not Required"
                fert_color = "#d4edda"           if rec_fert else "#f8d7da"
                st.markdown(
                    f"""<div style="background:{fert_color};border-radius:12px;
                                    padding:18px;text-align:center;">
                        <h4>🧪 Fertilizer</h4>
                        <h2>{fert_text}</h2>
                        <p>Based on historical data for<br>
                        <b>{inp_crop}</b> in <b>{inp_region}</b> on <b>{inp_soil}</b> soil.</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with rb:
                irri_text  = "✅ Recommended"   if rec_irri else "❌ Not Required"
                irri_color = "#d4edda"           if rec_irri else "#f8d7da"
                irri_note  = (
                    "💧 Low rainfall expected — irrigation strongly advised."   if low_rain  else
                    "🌧️ High rainfall expected — natural water may suffice."    if high_rain else
                    "Moderate rainfall — irrigate based on crop stage."
                )
                st.markdown(
                    f"""<div style="background:{irri_color};border-radius:12px;
                                    padding:18px;text-align:center;">
                        <h4>💧 Irrigation</h4>
                        <h2>{irri_text}</h2>
                        <p>{irri_note}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.success(
                f"📈 Expected Avg Yield with this combination: **{exp_yield:.2f} tons/hectare**"
            )

            harvest_est = sub["Days_to_Harvest"].mean()
            st.info(
                f"🗓️ Estimated Days to Harvest for **{inp_crop}** in **{inp_region}**: "
                f"**{harvest_est:.0f} days**"
            )

    st.markdown("---")

    # ── ROW 7 : Raw Data Table ────────────────────────────────────────────────
    with st.expander("📋 View Filtered Raw Data"):
        st.dataframe(fdf.reset_index(drop=True), use_container_width=True, height=300)
        st.download_button(
            "⬇️ Download Filtered Data as CSV",
            data=fdf.to_csv(index=False).encode("utf-8"),
            file_name="filtered_crop_yield.csv",
            mime="text/csv",
        )
