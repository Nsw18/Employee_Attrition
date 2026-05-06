import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# Import custom modules
from src.preprocessing import load_data, clean_data, preprocess_for_ml
from src.eda import (plot_attrition_distribution, plot_numerical_distribution, 
                     plot_categorical_vs_attrition, plot_correlation_heatmap, plot_box_bivariate)
from src.statistics import get_significant_features
from src.insight_engine import generate_insights
from src.ml_model import train_logistic_regression, get_feature_importance, generate_ml_insights
from src.report_generator import generate_pdf_report

# Setup Streamlit page configuration
st.set_page_config(page_title="HR Attrition Analytics", page_icon="📊", layout="wide")

# Title
st.title("📊 Employee Attrition Analysis & Insight Engine")
st.markdown("A data-driven approach to understanding and predicting employee attrition.")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
    ["1. Data Overview", "2. Exploratory Data Analysis", "3. Statistical Findings", "4. ML Insights", "5. Generate PDF Report"]
)

# Sidebar for data upload
st.sidebar.markdown("---")
st.sidebar.header("Dataset Configuration")
uploaded_file = st.sidebar.file_uploader("Upload HR Dataset (CSV)", type=["csv"])
default_data_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

# Make an images directory for saving plots
os.makedirs("images", exist_ok=True)

@st.cache_data
def load_and_prepare_data(file_or_path):
    df = load_data(file_or_path)
    if df is not None:
        df_clean = clean_data(df)
        df_ml = preprocess_for_ml(df_clean)
        return df, df_clean, df_ml
    return None, None, None

# Load data
df, df_clean, df_ml = None, None, None
if uploaded_file is not None:
    df, df_clean, df_ml = load_and_prepare_data(uploaded_file)
elif os.path.exists(default_data_path):
    df, df_clean, df_ml = load_and_prepare_data(default_data_path)
else:
    st.warning("Please upload a dataset or ensure the default dataset exists in the 'data' folder.")

if df_clean is not None:
    if page == "1. Data Overview":
        st.header("1. Data Overview")
        st.write("First 10 rows of the cleaned dataset:")
        st.dataframe(df_clean.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Info")
            st.write(f"Total Employees: {df_clean.shape[0]}")
            st.write(f"Total Features: {df_clean.shape[1]}")
            attrition_rate = (df_clean['Attrition'] == 'Yes').mean() * 100
            st.metric(label="Overall Attrition Rate", value=f"{attrition_rate:.2f}%")
            
        with col2:
            st.subheader("Data Types")
            dtypes = df_clean.dtypes.value_counts().reset_index()
            dtypes.columns = ['Data Type', 'Count']
            st.dataframe(dtypes)

    elif page == "2. Exploratory Data Analysis":
        st.header("2. Exploratory Data Analysis (EDA)")
        
        st.subheader("Overall Attrition Distribution")
        fig1 = plot_attrition_distribution(df_clean)
        st.pyplot(fig1)
        fig1.savefig("images/attrition_dist.png")
        
        st.subheader("Numerical Feature Distributions")
        num_col = st.selectbox("Select a numerical feature:", df_clean.select_dtypes(exclude=['object']).columns)
        fig2 = plot_numerical_distribution(df_clean, num_col)
        st.pyplot(fig2)
        
        st.subheader("Categorical Features vs Attrition")
        cat_col = st.selectbox("Select a categorical feature:", df_clean.select_dtypes(include=['object']).columns.drop('Attrition'))
        fig3 = plot_categorical_vs_attrition(df_clean, cat_col)
        st.pyplot(fig3)
        fig3.savefig("images/cat_vs_attrition.png")
        
        st.subheader("Correlation Heatmap")
        fig4 = plot_correlation_heatmap(df_clean)
        st.pyplot(fig4)
        fig4.savefig("images/heatmap.png")

    elif page == "3. Statistical Findings":
        st.header("3. Statistical Analysis & Insights")
        
        with st.spinner("Calculating statistics and generating insights..."):
            num_stats, cat_stats = get_significant_features(df_clean)
            insights = generate_insights(df_clean, num_stats, cat_stats)
            
        st.subheader("Automated Business Insights")
        for insight in insights:
            st.info(insight)
            
        st.subheader("Statistical Tests Results")
        tab1, tab2 = st.tabs(["Numerical Features (T-Test)", "Categorical Features (Chi-Square)"])
        
        with tab1:
            st.dataframe(num_stats)
            
        with tab2:
            st.dataframe(cat_stats)

    elif page == "4. ML Insights":
        st.header("4. Machine Learning Validation (Logistic Regression)")
        st.markdown("We use a simple Logistic Regression model to validate findings and extract interpretable coefficients.")
        
        with st.spinner("Training model..."):
            model, X_test, y_test, y_prob, accuracy, report, cm = train_logistic_regression(df_ml)
            feature_names = df_ml.drop(columns=['Attrition']).columns.tolist()
            importance_df = get_feature_importance(model, feature_names)
            ml_insights = generate_ml_insights(importance_df)
            
        st.success(f"Model trained successfully! Accuracy: {accuracy*100:.2f}%")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Key Drivers of Attrition")
            for ins in ml_insights:
                if ins.startswith("---"): continue
                if "Top Factors" in ins:
                    st.markdown(f"**{ins}**")
                else:
                    st.write(ins)
                    
        with col2:
            st.subheader("Top Coefficients Data")
            st.dataframe(importance_df.head(10))

    elif page == "5. Generate PDF Report":
        st.header("5. Generate Final PDF Report")
        st.write("Click the button below to compile the data, statistics, and machine learning insights into a professional PDF report.")
        
        if st.button("Generate Report"):
            with st.spinner("Compiling report..."):
                # Pre-calculate to pass to report generator
                num_stats, cat_stats = get_significant_features(df_clean)
                insights = generate_insights(df_clean, num_stats, cat_stats)
                
                model, _, _, _, _, _, _ = train_logistic_regression(df_ml)
                feature_names = df_ml.drop(columns=['Attrition']).columns.tolist()
                importance_df = get_feature_importance(model, feature_names)
                ml_insights = generate_ml_insights(importance_df)
                
                # Make sure some default images exist if the user hasn't visited EDA page
                if not os.path.exists("images/attrition_dist.png"):
                    plot_attrition_distribution(df_clean).savefig("images/attrition_dist.png")
                    plot_correlation_heatmap(df_clean).savefig("images/heatmap.png")
                    # Save a categorical one for example
                    plot_categorical_vs_attrition(df_clean, 'Department').savefig("images/cat_vs_attrition.png")
                    plt.close('all') # Clear memory
                
                image_paths = ["images/attrition_dist.png", "images/cat_vs_attrition.png", "images/heatmap.png"]
                
                success, msg = generate_pdf_report(insights, ml_insights, image_paths, "Employee_Attrition_Report.pdf")
                
            if success:
                st.success(msg)
                with open("Employee_Attrition_Report.pdf", "rb") as pdf_file:
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_file,
                        file_name="Employee_Attrition_Report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.error(msg)
