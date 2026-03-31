import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Cab Cancellation AI", layout="wide")

# -----------------------------
# PREMIUM UI CSS
# -----------------------------
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #181A2F, #242E49, #37415C);
}

/* Glass Cards */
.glass {
    background: rgba(36, 46, 73, 0.6);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 25px;

    box-shadow: 0 10px 30px rgba(24, 26, 47, 0.8);
    transition: 0.3s ease;
}

/* Hover Effect */
.glass:hover {
    transform: translateY(-8px);
    box-shadow: 0 15px 40px rgba(253, 164, 129, 0.6);
}

/* Title */
.title {
    font-size: 34px;
    font-weight: 700;
    color: #FDA481;
    text-align: center;
}

/* Subtitle */
.subtitle {
    color: #E1D4C2;
    text-align: center;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #B4182D, #54162B);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}

.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #FDA481, #B4182D);
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(36, 46, 73, 0.6);
    padding: 15px;
    border-radius: 12px;
}

/* Text */
h1, h2, h3, p {
    color: #FDA481;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="glass">
    <div class="title">🚕 Cab Cancellation Prediction System</div>
    <div class="subtitle">Predict ride cancellations using Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# FILE UPLOAD
# -----------------------------
file = st.file_uploader("Upload Dataset")

if file:
    df = pd.read_csv(file)

    st.markdown("### 📊 Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column (Cancellation)", df.columns)

    # -----------------------------
    # TRAIN MODEL
    # -----------------------------
    if st.button("Train Model"):

        with st.spinner("Training model..."):

            X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

        st.success(f"Model Accuracy: {acc:.2f}")

        # -----------------------------
        # METRICS
        # -----------------------------
        col1, col2 = st.columns(2)

        col1.metric("Accuracy", f"{acc:.2f}")
        col2.metric("Total Rows", df.shape[0])

        # -----------------------------
        # GRAPH 1: TARGET DISTRIBUTION
        # -----------------------------
        st.markdown("### 📊 Cancellation Distribution")

        fig1, ax1 = plt.subplots()
        df[target].value_counts().plot(kind='bar', ax=ax1)
        st.pyplot(fig1)

        # -----------------------------
        # GRAPH 2: CORRELATION HEATMAP
        # -----------------------------
        st.markdown("### 🔥 Feature Correlation")

        fig2, ax2 = plt.subplots(figsize=(8,5))
        sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

        # -----------------------------
        # GRAPH 3: CONFUSION MATRIX
        # -----------------------------
        st.markdown("### 📉 Confusion Matrix")

        cm = confusion_matrix(y_test, preds)

        fig3, ax3 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax3)
        st.pyplot(fig3)

        # -----------------------------
        # GRAPH 4: FEATURE IMPORTANCE
        # -----------------------------
        st.markdown("### ⭐ Feature Importance")

        importance = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(10)

        fig4, ax4 = plt.subplots()
        ax4.barh(feat_df["Feature"], feat_df["Importance"])
        ax4.invert_yaxis()
        st.pyplot(fig4)

        # -----------------------------
        # SAVE MODEL
        # -----------------------------
        st.session_state["model"] = model
        st.session_state["columns"] = X.columns

# -----------------------------
# PREDICTION SECTION
# -----------------------------
if "model" in st.session_state:

    st.markdown("""
    <div class="glass">
        <h3>🔮 Make Prediction</h3>
    </div>
    """, unsafe_allow_html=True)

    model = st.session_state["model"]
    cols = st.session_state["columns"]

    user_input = {}

    for col in cols[:5]:  # limit for UI
        user_input[col] = st.number_input(col, value=0.0)

    if st.button("Predict Cancellation"):

        input_df = pd.DataFrame([user_input])
        input_df = input_df.reindex(columns=cols, fill_value=0)

        pred = model.predict(input_df)[0]

        if pred == 1:
            st.error("🚨 Ride likely to CANCEL")
        else:
            st.success("✅ Ride likely to COMPLETE")
