import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def generate_visualizations(df, model_metrics, feature_importances):
    """
    Generate visualizations for the dataset using Matplotlib and convert to base64 strings.
    
    Args:
        df (DataFrame): Input DataFrame
        model_metrics (dict): Model performance metrics
        feature_importances (dict): Feature importance scores
        
    Returns:
        dict: Dictionary containing base64 encoded images of visualizations
    """
    visualizations = {}
    
    # Ensure we have an Attrition column (binary)
    if 'Attrition' in df.columns:
        # Convert attrition to binary format if needed
        if df['Attrition'].dtype == object:
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    else:
        return {"error": "No Attrition column found in dataset"}
    
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', vmin=-1, vmax=1, 
                square=True, linewidths=.5)
    plt.title('Feature Correlation Heatmap', fontsize=15)
    plt.tight_layout()
    visualizations['correlation_heatmap'] = _fig_to_base64(plt.gcf())
    plt.close()
    
    # 2. Attrition Distribution
    plt.figure(figsize=(10, 6))
    attrition_counts = df['Attrition'].value_counts()
    labels = ['Retained', 'Left']
    colors = ['#4CAF50', '#FF6B6B']
    plt.pie(attrition_counts, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    plt.title('Attrition Distribution', fontsize=15)
    plt.axis('equal')
    visualizations['attrition_distribution'] = _fig_to_base64(plt.gcf())
    plt.close()
    
    # 3. Top Features by Importance
    if feature_importances:
        # Sort features by importance
        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:15]
        feature_names = [item[0] for item in sorted_features]
        importance_values = [item[1] for item in sorted_features]
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
        bars = plt.barh(range(len(feature_names)), importance_values, color=colors)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Importance')
        plt.title('Top 15 Features by Importance', fontsize=15)
        plt.gca().invert_yaxis()  # Highest values at the top
        plt.tight_layout()
        visualizations['feature_importance'] = _fig_to_base64(plt.gcf())
        plt.close()
    
    # 4. Confusion Matrix
    if 'confusion_matrix' in model_metrics:
        conf_matrix = model_metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title('Confusion Matrix', fontsize=15)
        plt.tight_layout()
        visualizations['confusion_matrix'] = _fig_to_base64(plt.gcf())
        plt.close()
    
    # 5. Model Performance Metrics
    if model_metrics:
        metrics_to_plot = {k: v for k, v in model_metrics.items() 
                          if k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'] and v is not None}
        
        if metrics_to_plot:
            plt.figure(figsize=(10, 6))
            metrics_names = list(metrics_to_plot.keys())
            metrics_values = list(metrics_to_plot.values())
            
            # Create a bar chart
            bars = plt.bar(metrics_names, metrics_values, color='skyblue')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.ylim(0, 1.15)  # Add some space for the text
            plt.ylabel('Score')
            plt.title('Model Performance Metrics', fontsize=15)
            plt.tight_layout()
            visualizations['model_metrics'] = _fig_to_base64(plt.gcf())
            plt.close()
    
    # 6. ROC Curve
    if model_metrics and 'roc_curve' in model_metrics and model_metrics['roc_curve']:
        roc_data = model_metrics['roc_curve']
        fpr = roc_data.get('fpr')
        tpr = roc_data.get('tpr')
        
        if fpr is not None and tpr is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {model_metrics.get("roc_auc", 0):.3f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic', fontsize=15)
            plt.legend(loc="lower right")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            visualizations['roc_curve'] = _fig_to_base64(plt.gcf())
            plt.close()
    
    return visualizations

def _fig_to_base64(fig):
    """
    Convert a Matplotlib figure to a base64 encoded string.
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        str: Base64 encoded string of the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return img_str