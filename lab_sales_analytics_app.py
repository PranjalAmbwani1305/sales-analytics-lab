import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


st.set_page_config(page_title="Sales Forecast Visual Analytics Lab", layout="wide")
st.title("Sales Forecast Visual Analytics Lab")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("The dataset must contain at least two numeric columns.")
        st.stop()

    st.sidebar.header("Configuration")
    feature_cols = st.sidebar.multiselect("Select Feature Columns", numeric_cols)
    target_col = st.sidebar.selectbox("Select Target Column", numeric_cols)
    model_type = st.sidebar.selectbox("Select Model", ["Linear Regression", "Decision Tree"])
    test_size_pct = st.sidebar.slider("Test Set Size (%)", 10, 50, 20)

    if feature_cols and target_col and target_col not in feature_cols:
        clean_df = df[feature_cols + [target_col]].dropna()
        X = clean_df[feature_cols]
        y = clean_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_pct / 100, random_state=42
        )

        if model_type == "Linear Regression":
            model = LinearRegression()
        else:
            model = DecisionTreeRegressor(random_state=42, max_depth=5)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

  
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        st.sidebar.subheader("Model Performance")
        st.sidebar.write(f"R² Score: {r2:.3f}")
        st.sidebar.write(f"Mean Absolute Error: {mae:.2f}")
        st.sidebar.write(f"Root Mean Squared Error: {rmse:.2f}")

        result_df = X_test.copy()
        result_df[target_col] = y_test
        result_df["Predicted"] = predictions
        result_df["Error"] = result_df[target_col] - result_df["Predicted"]

        st.subheader("Actual vs Predicted Data")
        st.dataframe(result_df)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Actual vs Predicted Plot")
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            ax1.scatter(result_df[target_col], result_df["Predicted"], alpha=0.7, edgecolors='black')
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

        with col2:
            st.markdown("### Residual Analysis")
            residuals = y_test - predictions
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.scatter(predictions, residuals, alpha=0.6)
            ax2.axhline(0, color="red", linestyle="--")
            ax2.set_xlabel("Predicted Values")
            ax2.set_ylabel("Residuals")
            ax2.grid(True)
            st.pyplot(fig2)
        if hasattr(model, "coef_"):
            coef_df = pd.DataFrame({
                "Feature": feature_cols,
                "Coefficient": model.coef_,
                "Absolute Coefficient": np.abs(model.coef_)
            }).sort_values("Absolute Coefficient", ascending=False)

            st.subheader("Feature Coefficients")
            st.dataframe(coef_df[["Feature", "Coefficient"]])
        elif hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)
            st.subheader("Feature Importance (Decision Tree)")
            st.dataframe(importance_df)
            
        st.subheader("Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)

        st.subheader("Make a New Prediction")

        with st.form("custom_prediction"):
            st.write("Enter values for each selected feature:")
            custom_values = {}
            for col in feature_cols:
                val = st.number_input(f"{col}", value=float(X[col].mean()))
                custom_values[col] = val

            submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([custom_values])
            custom_prediction = model.predict(input_df)[0]
            st.success(f"Predicted {target_col}: {custom_prediction:.2f}")

            input_df[target_col] = custom_prediction
            st.download_button(
                "Download Prediction Result",
                input_df.to_csv(index=False),
                "custom_prediction.csv"
            )
        st.download_button("Download Model Predictions", result_df.to_csv(index=False), "sales_predictions.csv")

    else:
        st.warning("Please select at least one feature and a distinct target column.")

else:
    st.info("Upload a CSV file from the sidebar to begin your analysis.")
