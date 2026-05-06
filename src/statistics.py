import pandas as pd
import numpy as np
from scipy import stats

def calculate_numerical_statistics(df, numerical_cols, target_col='Attrition'):
    """
    Performs independent t-tests for numerical features to see if there's a 
    significant difference in means between Attrition classes.
    """
    results = []
    
    # Split data based on Attrition
    # Assumes target_col has been encoded to 1/0 or Yes/No. 
    # Let's handle both dynamically
    unique_vals = df[target_col].dropna().unique()
    if 'Yes' in unique_vals or 'No' in unique_vals:
        attrition_yes = df[df[target_col] == 'Yes']
        attrition_no = df[df[target_col] == 'No']
    elif 1 in unique_vals or 0 in unique_vals:
        attrition_yes = df[df[target_col] == 1]
        attrition_no = df[df[target_col] == 0]
    elif len(unique_vals) >= 2:
        attrition_yes = df[df[target_col] == unique_vals[0]]
        attrition_no = df[df[target_col] == unique_vals[1]]
    else:
        attrition_yes = df.iloc[0:0] # Empty
        attrition_no = df.iloc[0:0]

    for col in numerical_cols:
        # Ignore target if in numerical cols
        if col == target_col: continue
            
        data_yes = attrition_yes[col].dropna()
        data_no = attrition_no[col].dropna()
        
        if len(data_yes) == 0 or len(data_no) == 0:
            continue
            
        # Perform independent t-test
        t_stat, p_val = stats.ttest_ind(data_yes, data_no, equal_var=False)
        
        mean_yes = data_yes.mean()
        mean_no = data_no.mean()
        
        # Calculate Confidence Interval for the difference in means
        diff = mean_yes - mean_no
        se = np.sqrt(data_yes.var()/len(data_yes) + data_no.var()/len(data_no))
        margin = 1.96 * se # 95% CI
        ci_lower = diff - margin
        ci_upper = diff + margin
        
        results.append({
            'Feature': col,
            'Mean (Attrition=Yes)': round(mean_yes, 2),
            'Mean (Attrition=No)': round(mean_no, 2),
            'Difference': round(diff, 2),
            'P-Value': p_val,
            'Significant': 'Yes' if p_val < 0.05 else 'No',
            '95% CI': f"[{ci_lower:.2f}, {ci_upper:.2f}]"
        })
        
    return pd.DataFrame(results)

def calculate_categorical_statistics(df, categorical_cols, target_col='Attrition'):
    """
    Performs Chi-Square Test of Independence for categorical features.
    """
    results = []
    
    for col in categorical_cols:
        if col == target_col: continue
            
        # Create a contingency table
        contingency_table = pd.crosstab(df[col], df[target_col])
        
        # Perform Chi-Square test
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
        
        results.append({
            'Feature': col,
            'Chi2 Statistic': round(chi2, 2),
            'P-Value': p_val,
            'Significant': 'Yes' if p_val < 0.05 else 'No'
        })
        
    return pd.DataFrame(results)

def get_significant_features(df, target_col='Attrition'):
    """
    Wrapper function to get a list of features that are statistically significant.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    num_results = pd.DataFrame()
    cat_results = pd.DataFrame()
    
    if len(numerical_cols) > 0:
        num_results = calculate_numerical_statistics(df, numerical_cols, target_col)
        
    if len(categorical_cols) > 0:
        cat_results = calculate_categorical_statistics(df, categorical_cols, target_col)
        
    return num_results, cat_results
