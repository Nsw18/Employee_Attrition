import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    df_clean = df.copy()
    
    # Drop columns that have only one unique value or are just IDs (no predictive power)
    # EmployeeCount, Over18, StandardHours usually have 1 unique value in this dataset
    cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
    
    if existing_cols_to_drop:
        df_clean = df_clean.drop(columns=existing_cols_to_drop)
        
    # Handle missing values: Fill numerical with median, categorical with mode
    # (IBM dataset is usually clean, but this makes it robust)
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                
    return df_clean

def preprocess_for_ml(df):
    df_ml = df.copy()
    
    # Target variable encoding (Attrition: Yes=1, No=0)
    if 'Attrition' in df_ml.columns:
        df_ml['Attrition'] = df_ml['Attrition'].map({'Yes': 1, 'No': 0})
        
    # Separate categorical and numerical columns
    categorical_cols = df_ml.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df_ml.select_dtypes(exclude=['object']).columns.tolist()
    if 'Attrition' in numerical_cols:
        numerical_cols.remove('Attrition') # don't scale the target
        
    # Label Encoding for binary categoricals or ordinal features if necessary
    # For simplicity and interpretability in Logistic Regression, we use One-Hot Encoding for most
    binary_cols = [col for col in categorical_cols if df_ml[col].nunique() == 2]
    le = LabelEncoder()
    for col in binary_cols:
        df_ml[col] = le.fit_transform(df_ml[col])
        
    # Remove binary from categorical for one-hot
    categorical_cols = [c for c in categorical_cols if c not in binary_cols]
    
    # One-Hot Encoding for multi-class categoricals
    if categorical_cols:
        df_ml = pd.get_dummies(df_ml, columns=categorical_cols, drop_first=True)
        
    # Feature Scaling using StandardScaler for numericals
    scaler = StandardScaler()
    if numerical_cols:
        df_ml[numerical_cols] = scaler.fit_transform(df_ml[numerical_cols])
        
    return df_ml
