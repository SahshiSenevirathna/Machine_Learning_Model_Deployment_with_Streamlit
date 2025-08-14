# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Wine Quality Predictor", layout="wide")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path_model='../model.pkl', path_scaler='../scaler.pkl'):
    model = joblib.load(path_model)
    scaler = joblib.load(path_scaler)
    return model, scaler

st.title("Wine Quality Predictor")
st.write("Predict wine quality using a trained model. (Binary: good/bad)")

# Sidebar
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Overview", "Visualisations", "Model Prediction", "Model Performance", "About"])

# Load data + model
df = load_data('C:\Users\USER\OneDrive\Desktop\streamlit\WineQT.csv')
model, scaler = load_model(path_model='model.pkl', path_scaler='scaler.pkl')

if section == "Overview":
    st.header("Dataset Overview")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.dataframe(df.sample(10))
    st.markdown("**Filter**")
    col = st.selectbox("Filter column", df.columns)
    val = st.number_input("Filter value (numeric)", value=float(df[col].median()))
    st.dataframe(df[df[col] >= val].head(50))

elif section == "Visualisations":
    st.header("Visualisations")
    fig1 = px.histogram(df, x='quality', nbins=6, title='Quality Distribution')
    st.plotly_chart(fig1, use_container_width=True)

    corr = df.corr()
    fig2 = px.imshow(corr, text_auto=True, title='Correlation matrix')
    st.plotly_chart(fig2, use_container_width=True)

    # Interactive scatter
    xcol = st.selectbox("X axis", df.columns, index=0)
    ycol = st.selectbox("Y axis", df.columns, index=1)
    fig3 = px.scatter(df, x=xcol, y=ycol, color='quality')
    st.plotly_chart(fig3, use_container_width=True)

elif section == "Model Prediction":
    st.header("Make a prediction")
    st.write("Adjust feature values and click **Predict**.")
    features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
    user_input = {}
    cols = st.columns(2)
    for i, feat in enumerate(features):
        low = float(df[feat].min())
        high = float(df[feat].max())
        default = float(df[feat].median())
        user_input[feat] = cols[i%2].slider(feat, min_value=low, max_value=high, value=default)

    if st.button("Predict"):
        x = np.array([list(user_input.values())])
        x_scaled = scaler.transform(x)
        pred = model.predict(x_scaled)[0]
        st.success(f"Predicted class: {'Good (quality>=7)' if pred==1 else 'Bad (<7)'}")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_scaled)[0]
            st.write("Probability:", proba)

elif section == "Model Performance":
    st.header("Model Performance")
    # Evaluate on test split saved in notebook (you could save predictions & metrics to disk and load here)
    st.write("Load and show confusion matrix from evaluation (run notebook first to create eval file).")
    # Optionally compute here if dataset small: split and evaluate quickly
    from sklearn.model_selection import train_test_split
    
    X = df[features]
    y = (df['quality']>=7).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion matrix:")
    st.write(cm)
    st.text(classification_report(y_test, y_pred))

elif section == "About":
    st.header("About this app")
    st.markdown("""
    - Built with Streamlit
    - Model trained with scikit-learn
    - App includes EDA, visualization, prediction and performance
    """)
