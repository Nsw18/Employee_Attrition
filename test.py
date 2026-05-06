import os
import matplotlib.pyplot as plt

from src.preprocessing import load_data, clean_data, preprocess_for_ml
from src.eda import plot_attrition_distribution, plot_categorical_vs_attrition, plot_correlation_heatmap
from src.statistics import get_significant_features
from src.insight_engine import generate_insights
from src.ml_model import train_logistic_regression, get_feature_importance, generate_ml_insights
from src.report_generator import generate_pdf_report

def main():
    print("Loading data...")
    df = load_data('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    df_clean = clean_data(df)
    df_ml = preprocess_for_ml(df_clean)
    
    print("Generating EDA images...")
    os.makedirs("images", exist_ok=True)
    plot_attrition_distribution(df_clean).savefig("images/attrition_dist.png")
    plot_categorical_vs_attrition(df_clean, 'Department').savefig("images/cat_vs_attrition.png")
    plot_correlation_heatmap(df_clean).savefig("images/heatmap.png")
    plt.close('all')
    
    print("Running stats...")
    num_stats, cat_stats = get_significant_features(df_clean)
    insights = generate_insights(df_clean, num_stats, cat_stats)
    
    print("Running ML...")
    model, _, _, _, _, _, _ = train_logistic_regression(df_ml)
    feature_names = df_ml.drop(columns=['Attrition']).columns.tolist()
    importance_df = get_feature_importance(model, feature_names)
    ml_insights = generate_ml_insights(importance_df)
    
    print("Generating PDF...")
    image_paths = ["images/attrition_dist.png", "images/cat_vs_attrition.png", "images/heatmap.png"]
    success, msg = generate_pdf_report(insights, ml_insights, image_paths, "Test_Report.pdf")
    print(msg)

if __name__ == '__main__':
    main()
