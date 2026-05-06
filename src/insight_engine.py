import pandas as pd

def generate_insights(df, num_stats=None, cat_stats=None):
    """
    Generates human-readable business insights based on simple aggregations 
    and statistical significance.
    """
    insights = []
    
    # 1. Overall Attrition Rate
    total_employees = len(df)
    attrition_count = len(df[df['Attrition'] == 'Yes'])
    overall_rate = (attrition_count / total_employees) * 100
    insights.append(f"The overall employee attrition rate is {overall_rate:.2f}%.")
    
    # 2. Insights from Categorical features (e.g., Department, JobRole, OverTime)
    if 'OverTime' in df.columns:
        ot_yes = df[df['OverTime'] == 'Yes']
        ot_no = df[df['OverTime'] == 'No']
        
        ot_yes_attrition = len(ot_yes[ot_yes['Attrition'] == 'Yes']) / len(ot_yes) if len(ot_yes) > 0 else 0
        ot_no_attrition = len(ot_no[ot_no['Attrition'] == 'Yes']) / len(ot_no) if len(ot_no) > 0 else 0
        
        if ot_yes_attrition > ot_no_attrition:
            ratio = ot_yes_attrition / ot_no_attrition if ot_no_attrition > 0 else 0
            insights.append(f"Employees working OverTime are {ratio:.1f}x more likely to leave compared to those who do not.")
            
    if 'Department' in df.columns:
        dept_rates = df.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
        top_dept = dept_rates.index[0]
        top_rate = dept_rates.iloc[0]
        insights.append(f"The {top_dept} department has the highest attrition rate at {top_rate:.2f}%.")
        
    if 'BusinessTravel' in df.columns:
        travel_rates = df.groupby('BusinessTravel')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
        top_travel = travel_rates.index[0]
        insights.append(f"Employees who travel '{top_travel}' have the highest attrition rate ({travel_rates.iloc[0]:.2f}%).")

    # 3. Insights from Numerical features (e.g., MonthlyIncome, Age, TotalWorkingYears)
    if 'MonthlyIncome' in df.columns:
        avg_income_left = df[df['Attrition'] == 'Yes']['MonthlyIncome'].mean()
        avg_income_stayed = df[df['Attrition'] == 'No']['MonthlyIncome'].mean()
        if avg_income_left < avg_income_stayed:
            diff_pct = ((avg_income_stayed - avg_income_left) / avg_income_stayed) * 100
            insights.append(f"Employees who leave earn on average {diff_pct:.1f}% less per month (${avg_income_left:.2f} vs ${avg_income_stayed:.2f}).")
            
    if 'Age' in df.columns:
        avg_age_left = df[df['Attrition'] == 'Yes']['Age'].mean()
        avg_age_stayed = df[df['Attrition'] == 'No']['Age'].mean()
        insights.append(f"Employees who leave are generally younger (Avg Age: {avg_age_left:.1f} years) compared to those who stay ({avg_age_stayed:.1f} years).")

    # 4. Integrate statistical significance if provided
    if num_stats is not None and not num_stats.empty:
        significant_num = num_stats[num_stats['Significant'] == 'Yes']['Feature'].tolist()
        if significant_num:
            insights.append(f"Statistically significant numerical drivers of attrition include: {', '.join(significant_num[:5])}.")
            
    if cat_stats is not None and not cat_stats.empty:
        significant_cat = cat_stats[cat_stats['Significant'] == 'Yes']['Feature'].tolist()
        if significant_cat:
            insights.append(f"Statistically significant categorical drivers of attrition include: {', '.join(significant_cat[:5])}.")

    return insights
