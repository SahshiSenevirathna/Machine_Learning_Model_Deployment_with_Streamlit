import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load model & scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load dataset
df = pd.read_csv(r"C:\Users\USER\OneDrive\Desktop\streamlit\WineQT.csv")


# Sidebar Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Data Exploration", "Visualisation", "Prediction", "Model Performance"])

# Data Exploration
if menu == "Data Exploration":
    st.title("Wine Quality Dataset - Exploration")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")
    st.write(df.describe())
    st.write(df.dtypes)

# Visualisation
elif menu == "Visualisation":
    st.title("Wine Quality Data Visualisation")
    fig = px.histogram(df, x="quality", title="Wine Quality Distribution")
    st.plotly_chart(fig)
    corr = df.corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    st.plotly_chart(fig_corr)

# Prediction
elif menu == "Prediction":
    st.title("Wine Quality Prediction")
    inputs = []
    for col in df.columns[:-1]:  # exclude 'quality'
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        inputs.append(val)

    if st.button("Predict"):
        scaled_inputs = scaler.transform([inputs])
        prediction = model.predict(scaled_inputs)
        st.success(f"Predicted Wine Quality: {prediction[0]}")

# Model Performance
elif menu == "Model Performance":
    st.title("Model Performance Metrics")
    X = df.drop("quality", axis=1)
    y = df["quality"]
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)
