import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Streamlit app title
st.title("Heart Attack Prediction and Data Visualization")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())
    
    # Feature selection
    st.sidebar.write("## Feature Selection")
    target = st.sidebar.selectbox("Select the target column (Heart Attack)", list(df.columns))
    features = st.sidebar.multiselect("Select feature columns", list(df.columns), default=list(df.columns)[:-1])
    
    if target and features:
        X = df[features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### Model Accuracy: {accuracy:.2f}")
        
        # Visualization options
        st.sidebar.write("## Data Visualization")
        chart_type = st.sidebar.selectbox("Select a chart type", 
            ["Histogram", "Box Plot", "Scatter Plot", "Correlation Heatmap", "Pie Chart", "Bar Chart", "Pair Plot", "Violin Plot", "Line Chart"])
        
        st.write("### Data Visualization")
        
        if chart_type == "Histogram":
            col = st.selectbox("Select a column", df.columns)
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], kde=True)
            st.pyplot(plt)
        
        elif chart_type == "Box Plot":
            col = st.selectbox("Select a column", df.columns)
            plt.figure(figsize=(8, 5))
            sns.boxplot(y=df[col])
            st.pyplot(plt)
        
        elif chart_type == "Scatter Plot":
            x_col = st.selectbox("Select X-axis column", df.columns)
            y_col = st.selectbox("Select Y-axis column", df.columns)
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=df[x_col], y=df[y_col])
            st.pyplot(plt)
        
        elif chart_type == "Correlation Heatmap":
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot(plt)
        
        elif chart_type == "Pie Chart":
            col = st.selectbox("Select a categorical column", df.select_dtypes(include=['object']).columns)
            plt.figure(figsize=(6, 6))
            df[col].value_counts().plot.pie(autopct="%1.1f%%")
            st.pyplot(plt)
        
        elif chart_type == "Bar Chart":
            col = st.selectbox("Select a categorical column", df.select_dtypes(include=['object']).columns)
            plt.figure(figsize=(8, 5))
            sns.countplot(x=df[col])
            st.pyplot(plt)
        
        elif chart_type == "Pair Plot":
            st.write("Generating pair plot...")
            sns.pairplot(df)
            st.pyplot()
        
        elif chart_type == "Violin Plot":
            col = st.selectbox("Select a column", df.columns)
            plt.figure(figsize=(8, 5))
            sns.violinplot(y=df[col])
            st.pyplot(plt)
        
        elif chart_type == "Line Chart":
            x_col = st.selectbox("Select X-axis column", df.columns)
            y_col = st.selectbox("Select Y-axis column", df.columns)
            plt.figure(figsize=(8, 5))
            sns.lineplot(x=df[x_col], y=df[y_col])
            st.pyplot(plt)
