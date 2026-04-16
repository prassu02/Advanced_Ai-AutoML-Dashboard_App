import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pycaret.classification import setup as clf_setup, compare_models as clf_compare, finalize_model as clf_finalize, predict_model as clf_predict
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, finalize_model as reg_finalize, predict_model as reg_predict

st.set_page_config(page_title="AutoML Dashboard", layout="wide")

st.title("🚀 Universal AutoML Dashboard (Production Ready)")

# =============================
# FILE UPLOAD
# =============================
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    tab1, tab2, tab3 = st.tabs(["📊 EDA", "⚙️ Training", "📈 Results"])

    # =============================
    # 📊 TAB 1: EDA
    # =============================
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write("Shape:", df.shape)

        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum())

        col = st.selectbox("Select column for visualization", df.columns)

        fig, ax = plt.subplots()
        if df[col].dtype == 'object':
            df[col].value_counts().plot(kind='bar', ax=ax)
        else:
            df[col].plot(kind='hist', ax=ax)
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig2, ax2 = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax2)
        st.pyplot(fig2)

    # =============================
    # ⚙️ TAB 2: TRAINING
    # =============================
    with tab2:
        st.subheader("Model Training")

        target = st.selectbox("Select Target Column", df.columns)
        problem_type = st.radio("Problem Type", ["Classification", "Regression"])

        if st.button("Run AutoML"):

            try:
                # =============================
                # DATA CLEANING (CRITICAL)
                # =============================
                df_clean = df.copy()

                # Drop rows where target is missing
                df_clean = df_clean.dropna(subset=[target])

                # Fill numeric NaN
                df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

                # Fill categorical NaN
                for col in df_clean.select_dtypes(include='object'):
                    if df_clean[col].isnull().sum() > 0:
                        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

                # =============================
                # MODEL TRAINING
                # =============================
                with st.spinner("Training models..."):

                    if problem_type == "Classification":

                        # Check target classes
                        if df_clean[target].nunique() < 2:
                            st.error("❌ Target must have at least 2 classes")
                            st.stop()

                        clf_setup(data=df_clean, target=target, session_id=42, verbose=False)

                        best_model = clf_compare()

                        if best_model is None:
                            st.error("❌ No valid classification model found")
                            st.stop()

                        final_model = clf_finalize(best_model)

                        st.session_state["model"] = final_model
                        st.session_state["type"] = "clf"

                    else:

                        reg_setup(data=df_clean, target=target, session_id=42, verbose=False)

                        best_model = reg_compare()

                        if best_model is None:
                            st.error("❌ No valid regression model found")
                            st.stop()

                        final_model = reg_finalize(best_model)

                        st.session_state["model"] = final_model
                        st.session_state["type"] = "reg"

                    st.session_state["data"] = df_clean
                    st.session_state["target"] = target

                st.success("✅ Model training completed!")

            except Exception as e:
                st.error(f"❌ Error during training: {e}")

    # =============================
    # 📈 TAB 3: RESULTS
    # =============================
    with tab3:
        st.subheader("Results & Metrics")

        if "model" in st.session_state:

            model = st.session_state["model"]
            model_type = st.session_state["type"]
            df_clean = st.session_state["data"]
            target = st.session_state["target"]

            try:
                if model_type == "clf":

                    preds = clf_predict(model, data=df_clean)

                    st.write("### Predictions")
                    st.dataframe(preds.head())

                    # Accuracy
                    acc = (preds["Label"] == preds[target]).mean()
                    st.metric("Accuracy", round(acc, 3))

                    # Distribution
                    fig3, ax3 = plt.subplots()
                    preds["Label"].value_counts().plot(kind='bar', ax=ax3)
                    st.pyplot(fig3)

                else:

                    preds = reg_predict(model, data=df_clean)

                    st.write("### Predictions")
                    st.dataframe(preds.head())

                    # RMSE
                    rmse = ((preds[target] - preds["Label"])**2).mean() ** 0.5
                    st.metric("RMSE", round(rmse, 3))

                    # Scatter plot
                    fig4, ax4 = plt.subplots()
                    ax4.scatter(preds[target], preds["Label"])
                    ax4.set_xlabel("Actual")
                    ax4.set_ylabel("Predicted")
                    st.pyplot(fig4)

                # Download results
                csv = preds.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", csv, "results.csv", "text/csv")

            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")

        else:
            st.warning("⚠️ Train a model first in Training tab.")