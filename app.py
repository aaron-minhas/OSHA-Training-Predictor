import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="OSHA Training Predictor", layout="wide")
st.title("OSHA Training Deadline Predictor")
st.markdown("**Predict who will miss their safety training deadlines**")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("osha_model.joblib")

model = load_model()

# Load predictions if exist
try:
    df = pd.read_csv("predictions.csv")
    st.success("Loaded predictions for all employees!")
except:
    df = None
    st.warning("Run train_model.py first!")

if df is not None:
    st.subheader("All Training Assignments")

    # Create a styled version with proper risk highlighting
    def color_risk_score(val):
        if pd.isna(val):
            return ''
        try:
            val = float(val)
            if val > 70:
                color = '#ff4d4d'    # Bright red for high risk
            elif val > 40:
                color = '#ffaa00'    # Orange for medium
            else:
                color = '#90EE90'    # Light green for safe
            return f'background-color: {color}; color: black; font-weight: bold'
        except:
            return ''

    # Apply styling only to the Risk_Score_% column
    styled_df = df.style \
        .applymap(color_risk_score, subset=['Risk_Score_%']) \
        .format({'Risk_Score_%': '{:.1f}%'})

    st.dataframe(styled_df, use_container_width=True)

    # Metrics
    col1, col2, col3 = st.columns(3)
    total = len(df)
    high_risk = len(df[df['Risk_Score_%'] > 70])
    with col1:
        st.metric("Total Assignments", total)
    with col2:
        st.metric("HIGH RISK (>70%)", high_risk, delta=f"+{high_risk}")
    with col3:
        st.metric("Predicted Compliance Rate", f"{100 - (high_risk/total*100):.1f}%")

    # Download
    st.download_button(
        "Download Full Predictions (CSV)",
        data=df.to_csv(index=False).encode(),
        file_name="OSHA_Training_Compliance_Report.csv",
        mime="text/csv"
    )

# Sidebar — Manual prediction (FIXED VERSION)
st.sidebar.header("Quick Check One Person")
with st.sidebar.form("single_check"):
    name = st.text_input("Name", "Anthony Baxley")
    days_left = st.slider("Days until deadline", 0, 180, 22)
    total_courses = st.slider("Total courses assigned", 1, 50, 23)
    ceu = st.selectbox("Course CEU", [0.1, 1.0, 3.1], index=2)
    spanish = st.checkbox("Spanish course?", False)
    days_started = st.slider("Days since started", 0, 60, 8)

    if st.form_submit_button("Predict Risk"):
        progress = days_started / (days_started + days_left + 1)
        
        # THIS IS THE FIX: Create a DataFrame with correct column names
        input_df = pd.DataFrame([{
            'days_left': days_left,
            'days_since_start': days_started,
            'progress': progress,
            'is_spanish': int(spanish),
            'ceu': ceu,
            'total_courses': total_courses
        }])

        pred = model.predict(input_df)[0]
        risk = (1 - model.predict_proba(input_df)[0][1]) * 100

        st.sidebar.write(f"**{name}**")
        if pred == 1:
            st.sidebar.success(f"Will Finish On Time\nRisk: {risk:.1f}%")
        else:
            st.sidebar.error(f"HIGH RISK – Will Miss Deadline!\nRisk: {risk:.1f}%")
            st.balloons()

st.caption("Random Forest Model • Real-time OSHA Compliance Prediction • Built in 2025")