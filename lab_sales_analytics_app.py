import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# App Configuration
# ============================================================
st.set_page_config(page_title="Sales Forecast Visual Analytics Lab", layout="wide")
st.title("Sales Forecast Visual Analytics Lab")

# ============================================================
# File Upload
# ============================================================
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("The dataset must contain at least two numeric columns.")
        st.stop()

    # ============================================================
    # Sidebar Configuration
    # ============================================================
    st.sidebar.header("Configuration")
    feature_cols = st.sidebar.multiselect("Select Feature Columns", numeric_cols)
    target_col = st.sidebar.selectbox("Select Target Column", numeric_cols)
    test_size_pct = st.sidebar.slider("Test Set Size (%)", 10, 50, 20)

    # ============================================================
    # Data & Model Setup
    # ============================================================
    if feature_cols and target_col and target_col not in feature_cols:
        clean_df = df[feature_cols + [target_col]].dropna()
        X = clean_df[feature_cols]
        y = clean_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_pct / 100, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # ============================================================
        # Model Metrics
        # ============================================================
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        st.sidebar.subheader("Model Performance")
        st.sidebar.write(f"R² Score: {r2:.3f}")
        st.sidebar.write(f"Mean Absolute Error: {mae:.2f}")
        st.sidebar.write(f"Root Mean Squared Error: {rmse:.2f}")

        # ============================================================
        # Results DataFrame
        # ============================================================
        result_df = X_test.copy()
        result_df[target_col] = y_test
        result_df["Predicted"] = predictions
        result_df["Error"] = result_df[target_col] - result_df["Predicted"]

        st.subheader("Actual vs Predicted Data")
        st.dataframe(result_df)

        # ============================================================
        # Visual Analysis
        # ============================================================
        col1, col2 = st.columns(2)

        # Scatter Plot
        with col1:
            st.markdown("### Actual vs Predicted Plot")
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            ax1.scatter(result_df[target_col], result_df["Predicted"], edgecolors='black', alpha=0.7)
            lims = [min(result_df[target_col].min(), result_df["Predicted"].min()),
                    max(result_df[target_col].max(), result_df["Predicted"].max())]
            ax1.plot(lims, lims, 'r--', label='Ideal Line')
            ax1.set_xlim(lims)
            ax1.set_ylim(lims)
            ax1.set_xlabel("Actual Values")
            ax1.set_ylabel("Predicted Values")
            ax1.legend()
            ax1.grid(True)
            st.pyplot(fig1)

        # Residual Plot
        with col2:
            st.markdown("### Residual Analysis")
            residuals = y_test - predictions
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.scatter(predictions, residuals, alpha=0.6)
            ax2.axhline(0, color='red', linestyle='--')
            ax2.set_xlabel("Predicted Values")
            ax2.set_ylabel("Residuals")
            ax2.grid(True)
            st.pyplot(fig2)

        # ============================================================
        # Feature Importance
        # ============================================================
        if hasattr(model, "coef_"):
            coef_df = pd.DataFrame({
                "Feature": feature_cols,
                "Coefficient": model.coef_,
                "Absolute Coefficient": np.abs(model.coef_)
            }).sort_values("Absolute Coefficient", ascending=False)
            st.subheader("Feature Coefficients")
            st.dataframe(coef_df[["Feature", "Coefficient"]])

        # ============================================================
        # Correlation Heatmap
        # ============================================================
        st.subheader("Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)

        # ============================================================
        # Download Option
        # ============================================================
        st.download_button("Download Model Results", result_df.to_csv(index=False), "sales_forecast_results.csv")

    else:
        st.warning("Please select at least one feature and a distinct target column.")
else:
    st.info("Upload a CSV file from the sidebar to begin your analysis.")
