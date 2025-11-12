import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="Sales Forecast App", layout="wide")
st.title("Sales Forecasting & Data Quality Dashboard")

st.markdown("""
Welcome to the **Sales Forecasting System** — a complete data analysis, cleaning, and regression modeling tool.  
Upload your dataset in the sidebar to begin.
""")

# ==========================
# SIDEBAR - FILE UPLOAD
# ==========================
st.sidebar.header("Upload & Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    # ==========================
    # DATA CLEANING SETUP
    # ==========================
    st.sidebar.subheader("Data Cleaning Options")

    # Identify numeric & non-numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Handle missing values
    st.markdown("### Missing Value Summary")
    missing = df.isnull().sum()
    st.write(missing[missing > 0] if missing.sum() > 0 else "No missing values detected.")

    # Sidebar cleaning options
    handle_missing = st.sidebar.selectbox(
        "Handle Missing Values",
        ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"]
    )

    if handle_missing == "Drop Rows":
        df.dropna(inplace=True)
    elif handle_missing == "Fill with Mean":
        df.fillna(df.mean(numeric_only=True), inplace=True)
    elif handle_missing == "Fill with Median":
        df.fillna(df.median(numeric_only=True), inplace=True)
    elif handle_missing == "Fill with Mode":
        for col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Detect duplicate rows
    duplicates = df.duplicated().sum()
    st.markdown(f"### Duplicate Records: {duplicates}")
    if duplicates > 0:
        if st.sidebar.checkbox("Remove Duplicates"):
            df.drop_duplicates(inplace=True)
            st.success("Duplicate rows removed.")

    # ==========================
    # ERROR FINDING & QUALITY CHECKS
    # ==========================
    st.subheader("Data Quality & Error Analysis")

    # Detect negative or zero sales values (for domain checks)
    if "sales" in [c.lower() for c in df.columns]:
        sales_col = [c for c in df.columns if c.lower() == "sales"][0]
        invalid_sales = df[df[sales_col] <= 0]
        if not invalid_sales.empty:
            st.warning(f"Found {len(invalid_sales)} records with invalid or zero '{sales_col}' values.")
        else:
            st.success("No invalid sales values detected.")

    # Outlier detection (Z-score based)
    st.markdown("### Outlier Detection (Z-Score)")
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    outlier_count = (z_scores > 3).sum()
    st.write(outlier_count[outlier_count > 0] if (outlier_count > 0).any() else "No major outliers detected.")

    # ==========================
    # FEATURE & TARGET SELECTION
    # ==========================
    st.sidebar.subheader("Model Configuration")
    feature_cols = st.sidebar.multiselect("Select Feature Columns", numeric_cols)
    target_col = st.sidebar.selectbox("Select Target Column", numeric_cols)
    test_size_pct = st.sidebar.slider("Test Set Size (%)", 10, 50, 20)

    if feature_cols and target_col and target_col not in feature_cols:
        clean_df = df[feature_cols + [target_col]].dropna()
        X = clean_df[feature_cols]
        y = clean_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_pct / 100, random_state=42)

        # ==========================
        # MODEL TRAINING
        # ==========================
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # ==========================
        # MODEL PERFORMANCE
        # ==========================
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        st.sidebar.subheader("Model Performance")
        st.sidebar.metric("R² Score", f"{r2:.3f}")
        st.sidebar.metric("MAE", f"{mae:.2f}")
        st.sidebar.metric("RMSE", f"{rmse:.2f}")

        # ==========================
        # ACTUAL VS PREDICTED
        # ==========================
        st.subheader("Actual vs Predicted Comparison")
        result_df = X_test.copy()
        result_df[target_col] = y_test
        result_df["Predicted"] = predictions
        result_df["Error"] = result_df[target_col] - result_df["Predicted"]
        st.dataframe(result_df)

        # ==========================
        # VISUALIZATIONS
        # ==========================
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Actual vs Predicted Plot")
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            ax1.scatter(result_df[target_col], result_df["Predicted"], alpha=0.7, edgecolors="black")
            ax1.plot(
                [result_df[target_col].min(), result_df[target_col].max()],
                [result_df[target_col].min(), result_df[target_col].max()],
                "r--"
            )
            ax1.set_xlabel("Actual")
            ax1.set_ylabel("Predicted")
            ax1.grid(True)
            st.pyplot(fig1)

        with col2:
            st.markdown("#### Residual Plot")
            residuals = y_test - predictions
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.scatter(predictions, residuals, alpha=0.6)
            ax2.axhline(0, color="red", linestyle="--")
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Residuals")
            ax2.grid(True)
            st.pyplot(fig2)

        # ==========================
        # FEATURE IMPORTANCE
        # ==========================
        coef_df = pd.DataFrame({
            "Feature": feature_cols,
            "Coefficient": model.coef_,
            "Absolute Value": np.abs(model.coef_)
        }).sort_values("Absolute Value", ascending=False)

        st.subheader("Feature Importance (Coefficients)")
        st.dataframe(coef_df)

        # ==========================
        # CORRELATION HEATMAP
        # ==========================
        st.subheader("Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)

        # ==========================
        # CUSTOM PREDICTION
        # ==========================
        st.subheader("Custom Prediction Input")
        with st.form("custom_prediction_form"):
            custom_vals = {col: st.number_input(f"Enter {col}", value=float(X[col].mean())) for col in feature_cols}
            predict_btn = st.form_submit_button("Predict")

        if predict_btn:
            input_df = pd.DataFrame([custom_vals])
            custom_result = model.predict(input_df)[0]
            st.success(f"Predicted {target_col}: {custom_result:.2f}")
            input_df[target_col] = custom_result
            st.download_button("Download Custom Prediction", input_df.to_csv(index=False), "custom_prediction.csv")

        # ==========================
        # EXPORT RESULTS
        # ==========================
        st.download_button("Download Model Predictions", result_df.to_csv(index=False), "predictions.csv")

    else:
        st.warning("Please select valid feature(s) and target column (they cannot be the same).")
else:
    st.info("Upload a CSV file to begin.")
