import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="📊 Data Visualization & Analytics Lab", layout="wide")
st.title("📈 Sales Data Visualization & Forecasting Dashboard")

uploaded_file = st.file_uploader("📂 Upload Sales Dataset (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.write("**Shape of Data:**", df.shape)
    st.write("**Columns:**", list(df.columns))
    st.subheader("📋 Summary Statistics")
    st.write(df.describe())

    st.subheader("📊 Data Visualization")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Please upload a dataset with at least two numeric columns.")
        st.stop()

    x_col = st.selectbox("Select X-axis", numeric_cols)
    y_col = st.selectbox("Select Y-axis", numeric_cols)

    fig1 = px.scatter(df, x=x_col, y=y_col, color=df.columns[0], title=f"{y_col} vs {x_col}")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🔗 Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    cax = ax2.matshow(corr, cmap="coolwarm")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar(cax)
    st.pyplot(fig2)

    st.subheader("📈 Linear Regression Analysis")
    features = st.multiselect("Select Feature Columns (X)", numeric_cols)
    target = st.selectbox("Select Target Column (y)", numeric_cols)

    if features and target and target not in features:
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        st.write("### 📊 Model Performance")
        st.write(f"**R² Score:** {r2:.3f}")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")

        result_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})
        st.write("### 🧾 Actual vs Predicted Results")
        st.dataframe(result_df.head(20))

        fig3 = px.scatter(result_df, x="Actual", y="Predicted", title="Actual vs Predicted", color_discrete_sequence=["blue"])
        fig3.add_shape(type='line', x0=result_df["Actual"].min(), y0=result_df["Actual"].min(),
                       x1=result_df["Actual"].max(), y1=result_df["Actual"].max(), line=dict(color="red", dash="dash"))
        st.plotly_chart(fig3, use_container_width=True)

        st.download_button("📥 Download Predictions", result_df.to_csv(index=False), "predictions.csv")
    else:
        st.info("👉 Please select valid feature(s) and a target column.")
else:
    st.info("👈 Upload a CSV file to begin.")