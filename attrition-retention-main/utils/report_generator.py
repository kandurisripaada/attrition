import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

def generate_report(df, model_metrics=None, feature_importances=None):
    """
    Generate a basic HTML report for employee data analysis.
    
    Args:
        df (DataFrame): Input DataFrame with employee data
        model_metrics (dict, optional): Simple model performance metrics
        feature_importances (dict, optional): Feature importance scores
        
    Returns:
        str: HTML report content
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()
    
    # Generate simple data summary
    data_summary = _generate_data_summary(df)
    
    # Generate simple visualizations
    visualization_data = _generate_visualizations(df, feature_importances)
    
    # Generate simple recommendations
    recommendations = _generate_recommendations(feature_importances, df)
    
    # Create basic HTML report
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Employee Data Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .section {{ margin-bottom: 20px; padding: 15px; border-radius: 5px; background-color: #f8f9fa; }}
            .visualization {{ text-align: center; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .recommendation {{ background-color: #eaf5ea; padding: 10px; border-left: 5px solid #28a745; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Employee Data Analysis Report</h1>
            
            <div class="section">
                <h2>Data Overview</h2>
                {data_summary}
            </div>
            
            <div class="section">
                <h2>Key Insights</h2>
                
                <div class="visualization">
                    <h3>Department Distribution</h3>
                    <img src="data:image/png;base64,{visualization_data.get('department_dist', '')}" alt="Department Distribution">
                </div>
                
                <div class="visualization">
                    <h3>Job Satisfaction Distribution</h3>
                    <img src="data:image/png;base64,{visualization_data.get('satisfaction_dist', '')}" alt="Job Satisfaction Distribution">
                </div>
    """
    
    # Add feature importance section if available
    if feature_importances:
        html += f"""
                <div class="visualization">
                    <h3>Top Factors</h3>
                    <img src="data:image/png;base64,{visualization_data.get('feature_importance', '')}" alt="Feature Importance">
                </div>
        """
    
    html += f"""
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {recommendations}
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def _generate_data_summary(df):
    """Generate a basic summary of the dataset"""
    total_employees = len(df)
    
    # Calculate attrition rate if available
    attrition_rate = "N/A"
    if 'Attrition' in df.columns:
        attrition_count = df['Attrition'].map({'Yes': 1, 'No': 0}).sum() if df['Attrition'].dtype == object else df['Attrition'].sum()
        attrition_rate = f"{(attrition_count / total_employees) * 100:.1f}%" if total_employees > 0 else "N/A"
    
    # Generate basic HTML summary
    html = f"""
    <p><strong>Total Employees:</strong> {total_employees}</p>
    <p><strong>Attrition Rate:</strong> {attrition_rate}</p>
    
    <h3>Dataset Columns</h3>
    <table>
        <tr>
            <th>Column</th>
            <th>Type</th>
            <th>Missing Values</th>
        </tr>
    """
    
    for column in df.columns[:10]:  # Limit to first 10 columns for simplicity
        col_type = str(df[column].dtype)
        missing = df[column].isnull().sum()
        
        html += f"""
        <tr>
            <td>{column}</td>
            <td>{col_type}</td>
            <td>{missing}</td>
        </tr>
        """
    
    if len(df.columns) > 10:
        html += f"""
        <tr>
            <td colspan="3">... and {len(df.columns) - 10} more columns</td>
        </tr>
        """
    
    html += "</table>"
    return html

def _generate_visualizations(df, feature_importances=None):
    """Generate basic visualizations for the report"""
    visualizations = {}
    
    # 1. Department distribution if available
    if 'Department' in df.columns:
        plt.figure(figsize=(8, 5))
        df['Department'].value_counts().plot(kind='bar', color='skyblue')
        plt.title('Employee Distribution by Department')
        plt.ylabel('Number of Employees')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualizations['department_dist'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    
    # 2. Job satisfaction distribution if available
    if 'JobSatisfaction' in df.columns:
        plt.figure(figsize=(8, 5))
        df['JobSatisfaction'].value_counts().sort_index().plot(kind='bar', color='lightgreen')
        plt.title('Job Satisfaction Distribution')
        plt.xlabel('Satisfaction Level')
        plt.ylabel('Number of Employees')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualizations['satisfaction_dist'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    
    # 3. Feature importance if available
    if feature_importances:
        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
        feature_names = [item[0] for item in sorted_features]
        importance_values = [item[1] for item in sorted_features]
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importance_values, color='lightblue')
        plt.title('Top 10 Important Factors')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        visualizations['feature_importance'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    
    return visualizations

def _generate_recommendations(feature_importances, df):
    """Generate simple recommendations based on data"""
    # Basic recommendations
    recommendations = [
        "Review compensation policies to ensure they remain competitive in the industry.",
        "Implement flexible work arrangements to improve work-life balance.",
        "Create clear career development paths for employees.",
        "Establish a recognition program to acknowledge employee contributions.",
        "Conduct regular check-ins to discuss job satisfaction and career goals."
    ]
    
    # Add feature-specific recommendations if available
    if feature_importances:
        top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for feature, _ in top_features:
            if "Overtime" in feature:
                recommendations.append("Review workload distribution to prevent employee burnout.")
            elif "Income" in feature or "Salary" in feature:
                recommendations.append("Conduct a salary survey to ensure compensation is competitive.")
            elif "Satisfaction" in feature:
                recommendations.append("Implement regular job satisfaction surveys and address issues promptly.")
    
    # Format recommendations as HTML
    html = "<ul>"
    for rec in recommendations[:5]:  # Limit to 5 recommendations
        html += f"<li class='recommendation'>{rec}</li>"
    html += "</ul>"
    
    return html