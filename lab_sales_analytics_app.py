import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================================
# APP CONFIG
# ==========================================
st.set_page_config(page_title="Sales Forecast App", layout="wide")
st.title("Sales Forecasting Dashboard")

# ==========================================
# SIDEBAR - FILE UPLOAD
# ==========================================
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # ==========================================
        # INTERNAL DATA CLEANING (invisible to user)
        # ==========================================
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.drop_duplicates(inplace=True)
        df.dropna(how="all", inplace=True)
        df.fillna(df.mean(numeric_only=True), inplace=True)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("The dataset must contain at least two numeric columns.")
            st.stop()

        # ==========================================
        # SIDEBAR CONFIGURATION
        # ==========================================
        st.sidebar.header("Configuration")
        feature_cols = st.sidebar.multiselect("Select Feature Columns", numeric_cols)
        target_col = st.sidebar.selectbox("Select Target Column", numeric_cols)
        test_size_pct = st.sidebar.slider("Test Set Size (%)", 10, 50, 20)

        # ==========================================
        # VALIDATION CHECKS
        # ==========================================
        if not feature_cols:
            st.warning("Please select at least one feature column.")
        elif target_col in feature_cols:
            st.warning("Target column cannot be the same as feature column.")
        else:
            # ==========================================
            # TRAIN / TEST SPLIT
            # ==========================================
            clean_df = df[feature_cols + [target_col]].dropna()
            X = clean_df[feature_cols]
            y = clean_df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_pct / 100, random_state=42
            )

            # ==========================================
            # MODEL TRAINING
            # ==========================================
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # ==========================================
            # MODEL METRICS
            # ==========================================
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)

            st.sidebar.header("Model Performance")
            st.sidebar.metric("R² Score", f"{r2:.3f}")
            st.sidebar.metric("MAE", f"{mae:.2f}")
            st.sidebar.metric("RMSE", f"{rmse:.2f}")

            # ==========================================
            # RESULTS TABLE
            # ==========================================
            result_df = X_test.copy()
            result_df[target_col] = y_test
            result_df["Predicted"] = predictions
            result_df["Error"] = result_df[target_col] - result_df["Predicted"]

            st.subheader("Actual vs Predicted Data")
            st.dataframe(result_df)

            # ==========================================
            # SCATTER PLOT
            # ==========================================
            st.subheader("Actual vs Predicted Plot")
            fig1, ax1 = plt.subplots(figsize=(7, 5))
            ax1.scatter(result_df[target_col], result_df["Predicted"], alpha=0.7, edgecolors="black")
            ax1.plot(
                [result_df[target_col].min(), result_df[target_col].max()],
                [result_df[target_col].min(), result_df[target_col].max()],
                "r--"
            )
            ax1.set_xlabel("Actual")
            ax1.set_ylabel("Predicted")
            ax1.set_title("Actual vs Predicted")
            ax1.grid(True)
            st.pyplot(fig1)

            # ==========================================
            # RESIDUAL PLOT
            # ==========================================
            st.subheader("Residual Analysis")
            residuals = y_test - predictions
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.scatter(predictions, residuals, alpha=0.6)
            ax2.axhline(0, color="red", linestyle="--")
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Residuals")
            ax2.set_title("Residual Plot")
            ax2.grid(True)
            st.pyplot(fig2)

            # ==========================================
            # FEATURE IMPORTANCE
            # ==========================================
            coef_df = pd.DataFrame({
                "Feature": feature_cols,
                "Coefficient": model.coef_,
                "Absolute Value": np.abs(model.coef_)
            }).sort_values("Absolute Value", ascending=False)

            st.subheader("Feature Coefficients")
            st.dataframe(coef_df[["Feature", "Coefficient"]])

            # ==========================================
            # CORRELATION HEATMAP
            # ==========================================
            st.subheader("Correlation Heatmap")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax3)
            st.pyplot(fig3)

            # ==========================================
            # DOWNLOAD OPTION
            # ==========================================
            st.download_button("Download Full Results", result_df.to_csv(index=False), "model_predictions.csv")

    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")

else:
    st.info("Upload a CSV file from the sidebar to start analysis.")
