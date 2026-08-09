"""
EDA Visualization Generator - Creates professional charts
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import base64
from io import BytesIO
from typing import Dict, List, Any

sns.set_style('whitegrid')
plt.rcParams['figure.facecolor'] = 'white'


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


def generate_distribution_plots(df: pd.DataFrame, target: str) -> List[Dict[str, str]]:
    """Generate distribution plots for numeric features"""
    plots = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    
    # Limit to first 6 for performance
    for col in numeric_cols[:6]:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histogram with KDE
            axes[0].hist(df[col].dropna(), bins=30, edgecolor='black', 
                        alpha=0.7, color='#6366f1', density=True)
            df[col].dropna().plot(kind='kde', ax=axes[0], color='#ef4444', linewidth=2)
            axes[0].set_title(f'{col} - Distribution', fontsize=12, fontweight='bold')
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Density')
            axes[0].grid(True, alpha=0.3)
            
            # Box plot
            axes[1].boxplot(df[col].dropna(), vert=True)
            axes[1].set_title(f'{col} - Box Plot', fontsize=12, fontweight='bold')
            axes[1].set_ylabel(col)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots.append({
                "title": f"Distribution Analysis: {col}",
                "type": "distribution",
                "image": fig_to_base64(fig)
            })
        except Exception as e:
            print(f"Error plotting {col}: {e}")
            continue
    
    return plots



def generate_correlation_heatmap(df: pd.DataFrame, target: str) -> Dict[str, str]:
    """Generate correlation heatmap"""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return None
        
        # Limit to top 15 features for readability
        if len(numeric_df.columns) > 15:
            # Get features most correlated with target
            if target in numeric_df.columns:
                corr_with_target = numeric_df.corr()[target].abs().sort_values(ascending=False)
                top_features = corr_with_target.head(15).index.tolist()
                numeric_df = numeric_df[top_features]
        
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return {
            "title": "Correlation Heatmap",
            "type": "correlation",
            "image": fig_to_base64(fig)
        }
    except Exception as e:
        print(f"Error generating correlation heatmap: {e}")
        return None


def generate_target_analysis(df: pd.DataFrame, target: str, task_type: str) -> Dict[str, str]:
    """Generate target variable analysis"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if task_type == "classification":
            # Count plot
            value_counts = df[target].value_counts()
            axes[0].bar(range(len(value_counts)), value_counts.values, 
                       color='#10b981', edgecolor='black', alpha=0.8)
            axes[0].set_xticks(range(len(value_counts)))
            axes[0].set_xticklabels(value_counts.index, rotation=45)
            axes[0].set_title(f'Target Distribution: {target}', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Count')
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Pie chart
            axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                       startangle=90, colors=sns.color_palette('Set2'))
            axes[1].set_title(f'Target Proportions', fontsize=12, fontweight='bold')
        else:
            # Histogram
            axes[0].hist(df[target].dropna(), bins=30, edgecolor='black',
                        alpha=0.7, color='#6366f1')
            axes[0].set_title(f'Target Distribution: {target}', fontsize=12, fontweight='bold')
            axes[0].set_xlabel(target)
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
            
            # Box plot
            axes[1].boxplot(df[target].dropna(), vert=True)
            axes[1].set_title(f'Target Box Plot', fontsize=12, fontweight='bold')
            axes[1].set_ylabel(target)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return {
            "title": f"Target Analysis: {target}",
            "type": "target",
            "image": fig_to_base64(fig)
        }
    except Exception as e:
        print(f"Error generating target analysis: {e}")
        return None



def generate_missing_values_plot(df: pd.DataFrame) -> Dict[str, str]:
    """Generate missing values visualization"""
    try:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            return None
        
        missing_pct = (missing / len(df) * 100)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(missing)), missing_pct.values, color='#ef4444', alpha=0.8)
        ax.set_yticks(range(len(missing)))
        ax.set_yticklabels(missing.index)
        ax.set_xlabel('Missing Percentage (%)', fontsize=11)
        ax.set_title('Missing Values Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, (idx, val) in enumerate(zip(missing.index, missing_pct.values)):
            ax.text(val + 0.5, i, f'{val:.1f}%', va='center')
        
        plt.tight_layout()
        
        return {
            "title": "Missing Values",
            "type": "missing",
            "image": fig_to_base64(fig)
        }
    except Exception as e:
        print(f"Error generating missing values plot: {e}")
        return None


def generate_comprehensive_eda(df: pd.DataFrame, target: str, task_type: str) -> List[Dict[str, str]]:
    """Generate all EDA visualizations"""
    visualizations = []
    
    # 1. Distribution plots for numeric features
    print("Generating distribution plots...")
    dist_plots = generate_distribution_plots(df, target)
    visualizations.extend(dist_plots)
    
    # 2. Correlation heatmap
    print("Generating correlation heatmap...")
    corr_plot = generate_correlation_heatmap(df, target)
    if corr_plot:
        visualizations.append(corr_plot)
    
    # 3. Target analysis
    print("Generating target analysis...")
    target_plot = generate_target_analysis(df, target, task_type)
    if target_plot:
        visualizations.append(target_plot)
    
    # 4. Missing values
    print("Generating missing values plot...")
    missing_plot = generate_missing_values_plot(df)
    if missing_plot:
        visualizations.append(missing_plot)
    
    print(f"Generated {len(visualizations)} visualizations")
    return visualizations
