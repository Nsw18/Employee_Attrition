import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_attrition_distribution(df):
    """
    Plots the overall distribution of the Attrition target variable.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Attrition', palette='Set2', ax=ax)
    ax.set_title('Distribution of Employee Attrition')
    ax.set_ylabel('Count')
    
    # Add value counts on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5), 
                    textcoords='offset points')
    plt.tight_layout()
    return fig

def plot_numerical_distribution(df, column, hue='Attrition'):
    """
    Plots the distribution of a numerical column, separated by Attrition.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x=column, hue=hue, kde=True, palette='Set1', ax=ax, bins=30)
    ax.set_title(f'Distribution of {column} by {hue}')
    plt.tight_layout()
    return fig

def plot_categorical_vs_attrition(df, column):
    """
    Plots a categorical variable against Attrition to show proportions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a cross-tabulation of the categorical feature vs Attrition
    ct = pd.crosstab(df[column], df['Attrition'], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, color=['#1f77b4', '#d62728'], ax=ax)
    
    ax.set_title(f'Attrition Rate by {column}')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel(column)
    plt.xticks(rotation=45, ha='right')
    
    # Legend
    plt.legend(title='Attrition', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap for numerical variables.
    """
    fig, ax = plt.subplots(figsize=(25, 25))
    
    # Compute correlation matrix for numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    # Draw the heatmap
    sns.heatmap(corr, cmap='coolwarm', vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax, annot=True)
    
    ax.set_title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    return fig

def plot_box_bivariate(df, num_col, cat_col='Attrition'):
    """
    Creates a boxplot for a numerical variable vs categorical (Attrition).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=cat_col, y=num_col, data=df, palette='Set3', ax=ax)
    ax.set_title(f'Boxplot of {num_col} by {cat_col}')
    plt.tight_layout()
    return fig
