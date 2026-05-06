import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_logistic_regression(df_ml, target_col='Attrition'):
    """
    Trains a Logistic Regression model on the preprocessed dataset.
    Returns the trained model, test data, and performance metrics.
    """
    X = df_ml.drop(columns=[target_col])
    y = df_ml[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and train model
    # Using liblinear solver for small datasets and L2 penalty
    model = LogisticRegression(solver='liblinear', random_state=42, max_iter=500)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, X_test, y_test, y_prob, accuracy, report, cm

def get_feature_importance(model, feature_names):
    """
    Extracts the coefficients from the Logistic Regression model to explain feature impact.
    """
    coefs = model.coef_[0]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs
    })
    
    # Calculate Odds Ratio (exp(coef)) to interpret the impact
    importance_df['Odds_Ratio'] = np.exp(importance_df['Coefficient'])
    
    # Sort by absolute coefficient to find the most impactful features (positive or negative)
    importance_df['Abs_Coefficient'] = importance_df['Coefficient'].abs()
    importance_df = importance_df.sort_values(by='Abs_Coefficient', ascending=False).drop(columns=['Abs_Coefficient'])
    
    return importance_df

def generate_ml_insights(importance_df, top_n=5):
    """
    Translates top coefficients into readable insights.
    """
    insights = []
    
    top_pos = importance_df[importance_df['Coefficient'] > 0].head(top_n)
    top_neg = importance_df[importance_df['Coefficient'] < 0].head(top_n)
    
    insights.append("--- ML Drivers of Attrition (Logistic Regression) ---")
    
    insights.append("\nTop Factors INCREASING the probability of Attrition:")
    for _, row in top_pos.iterrows():
        feat = row['Feature']
        odds = row['Odds_Ratio']
        insights.append(f"- {feat}: Increases odds of leaving by {odds:.2f}x")
        
    insights.append("\nTop Factors DECREASING the probability of Attrition (Retention Drivers):")
    for _, row in top_neg.iterrows():
        feat = row['Feature']
        odds = row['Odds_Ratio']
        # For negative coefficients, the Odds ratio is < 1, 
        # so we can say it reduces the odds by (1 - odds) * 100 %
        reduction = (1 - odds) * 100
        insights.append(f"- {feat}: Reduces odds of leaving by {reduction:.1f}%")
        
    return insights
