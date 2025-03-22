import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file, flash
from werkzeug.utils import secure_filename
import json
import joblib

from utils.data_processing import process_data
from utils.model_training import train_attrition_model
from utils.visualization import generate_visualizations
from utils.report_generator import generate_report
from utils.ai_utils import generate_retention_strategies

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.json.encoder = NumpyEncoder

# Global variables
df_memory = None
model_memory = None
feature_columns = []
categorical_columns = {}
numeric_ranges = {}
scaler = None
label_encoders = {}
model_metrics = {}
feature_importances = {}


@app.route('/')
@app.route('/reset')
def home():
    # Reset state only if accessing /reset endpoint
    if request.path == '/reset':
        # Clear relevant session data
        session_keys_to_clear = [
            'data_preview', 
            'feature_columns', 
            'categorical_columns', 
            'numeric_ranges',
            'accuracy', 
            'data_uploaded',
            'uploaded_filename',
            'clear_form_data'
        ]
        
        for key in session_keys_to_clear:
            if key in session:
                session.pop(key, None)
        
        # Reset global variables
        global df_memory, model_memory, feature_columns
        global categorical_columns, numeric_ranges, scaler
        global label_encoders, model_metrics, feature_importances
        
        df_memory = None
        model_memory = None
        feature_columns = []
        categorical_columns = {}
        numeric_ranges = {}
        scaler = None
        label_encoders = {}
        model_metrics = {}
        feature_importances = {}
        
        # Redirect to home after reset to clear the /reset from URL
        return redirect(url_for('home'))
    
    return render_template('home.html')

# Add this function to your app.py file to clear localStorage when a new dataset is uploaded
# Place it inside the upload_file function, right before the return statement

@app.route('/upload', methods=['POST'])
def upload_file():
    global df_memory, feature_columns, categorical_columns, numeric_ranges
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Store the uploaded filename in session
    session['uploaded_filename'] = secure_filename(file.filename)
    
    try:
        df_memory = pd.read_csv(file)
        
        # Convert 'Attrition' to binary format if it's not already (Yes/No -> 1/0)
        if 'Attrition' in df_memory.columns and df_memory['Attrition'].dtype == object:
            df_memory['Attrition'] = df_memory['Attrition'].map({'Yes': 1, 'No': 0})
        
        if 'Attrition' not in df_memory.columns:
            return jsonify({"error": "Dataset must contain an 'Attrition' column"}), 400
        
        feature_columns = [col for col in df_memory.columns if col != 'Attrition']
        
        # Store categorical columns for form rendering
        categorical_columns = {
            col: df_memory[col].dropna().unique().tolist()
            for col in df_memory.select_dtypes(include=['object']).columns
        }
        
        # Store numeric ranges for form rendering
        numeric_ranges = {
            col: {"min": float(df_memory[col].min()), "max": float(df_memory[col].max())}
            for col in df_memory.select_dtypes(include=[np.number]).columns
        }
        
        # Convert data preview to native Python types
        preview_data = df_memory.head(10).to_dict(orient='records')
        for row in preview_data:
            for key, value in row.items():
                if isinstance(value, (np.integer, np.floating)):
                    row[key] = float(value) if isinstance(value, np.floating) else int(value)
        
        session['data_preview'] = preview_data
        session['feature_columns'] = feature_columns
        session['categorical_columns'] = categorical_columns
        session['numeric_ranges'] = numeric_ranges
        session['data_uploaded'] = True
        
        # Instead of redirecting, return a JSON success response
        # This allows the AJAX call to handle the page refresh
        return jsonify({"success": True, "message": f"Successfully uploaded file: {session['uploaded_filename']}"})
    
    except Exception as e:
        # Handle any errors during file processing
        return jsonify({"error": str(e)}), 500
    


@app.route('/eda_train')
def eda_train():
    # Check if we have any flash messages from redirects
    
    # Add dataset status information
    dataset_status = {
        'uploaded': session.get('data_uploaded', False),
        'filename': session.get('uploaded_filename', None),
    }
    
    # Render the template
    response = render_template('eda_train.html',
                           data_preview=session.get('data_preview', []),
                           feature_columns=session.get('feature_columns', []),
                           categorical_columns=session.get('categorical_columns', {}),
                           numeric_ranges=session.get('numeric_ranges', {}),
                           accuracy=session.get('accuracy', None),
                           dataset_status=dataset_status)
    
    # Reset the clear_form_data flag after the template has been rendered
    if 'clear_form_data' in session:
        session.pop('clear_form_data', None)
    
    return response

@app.route('/analyze', methods=['POST'])
def analyze():
    global df_memory, label_encoders
    if df_memory is None:
        return jsonify({"error": "No file uploaded"}), 400
    
    df_memory, label_encoders = process_data(df_memory)
    
    # Convert data preview to native Python types
    preview_data = df_memory.head(10).to_dict(orient='records')
    for row in preview_data:
        for key, value in row.items():
            if isinstance(value, (np.integer, np.floating)):
                row[key] = float(value) if isinstance(value, np.floating) else int(value)
    
    session['data_preview'] = preview_data
    
    return jsonify({"message": "Data processing completed successfully!"})

@app.route('/train', methods=['POST'])
def train_model():
    global df_memory, model_memory, model_metrics, feature_importances, scaler
    if df_memory is None:
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        model_choice = request.form.get('model_choice')
        if model_choice not in ['logistic', 'random_forest', 'decision_tree']:
            return jsonify({"error": "Invalid model selection"}), 400
        
        model_memory, scaler, metrics, importances = train_attrition_model(df_memory, model_choice)
        model_metrics = metrics
        feature_importances = importances
        
        session['accuracy'] = metrics['accuracy']
        
        return jsonify({
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1_score": metrics['f1_score']
        })
    
    except Exception as e:
        print(f"Training Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict_page', methods=['GET'])
def predict_page():
    # Check if data is available
    global df_memory
    if df_memory is None:
        # Flash a message to inform the user
        flash('No dataset uploaded. Please upload data first.', 'warning')
        # Redirect to the upload/train page
        return redirect(url_for('eda_train'))
    
    # Check if model is available
    global model_memory
    if model_memory is None:
        # Flash a message to inform the user
        flash('No trained model available. Please upload data and train a model first.', 'warning')
        # Redirect to the upload/train page
        return redirect(url_for('eda_train'))
    
    # If both data and model are available, continue with normal rendering
    return render_template('predict.html',
                           feature_columns=session.get('feature_columns', []),
                           categorical_columns=session.get('categorical_columns', {}),
                           numeric_ranges=session.get('numeric_ranges', {}))

# Add this new route to handle direct navigation to '/predict'
@app.route('/predict', methods=['GET'])
def predict_redirect():
    return redirect(url_for('predict_page'))

@app.route('/predict', methods=['POST'])
def predict():
    global model_memory, scaler, label_encoders
    if model_memory is None:
        return jsonify({"error": "No trained model found!"}), 400
    
    # Collect input data from form
    input_data = {col: request.form.get(col) for col in session.get('feature_columns', [])}
    
    # Convert string values to appropriate types
    for col, value in input_data.items():
        if col in numeric_ranges:  # Numeric column
            input_data[col] = float(value) if value else 0
    
    # Create DataFrame from input
    df_input = pd.DataFrame([input_data])
    
    # Process user input data the same way as training data
    df_input, _ = process_data(df_input, label_encoders)
    
    # Scale the input data with the same scaler used for training
    if scaler is not None:
        input_scaled = scaler.transform(df_input)
    else:
        input_scaled = df_input.values
    
    # Make prediction
    prediction = model_memory.predict(input_scaled)[0]
    
    # Get probability if available
    if hasattr(model_memory, "predict_proba"):
        probability = model_memory.predict_proba(input_scaled)[0][1]  # Probability of class 1
    else:
        probability = None
    
    # Format the result
    result = {
        "prediction": int(prediction),
        "prediction_text": "Yes" if prediction == 1 else "No",
        "probability": float(probability) if probability is not None else None
    }
    
    # Generate retention strategies if the prediction is "Yes" (high attrition risk)
    if prediction == 1:
        result["retention_strategies"] = generate_retention_strategies(input_data)
    
    return jsonify(result)

@app.route('/visualize')
def visualize():
    global df_memory, model_metrics, feature_importances
    if df_memory is None:
        flash('No data available for visualization. Please upload data first.', 'warning')
        return redirect(url_for('eda_train'))
    
    visualizations = generate_visualizations(df_memory, model_metrics, feature_importances)
    return render_template('visualize.html', visualizations=visualizations)

@app.route('/download_model')
def download_model():
    model_path = 'models/attrition_model.pkl'
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({"error": "Model file not found"}), 404

@app.route('/download_report')
def download_report():
    global df_memory, model_metrics, feature_importances
    if df_memory is None:
        return jsonify({"error": "No data available for report generation"}), 400
    
    report_html = generate_report(df_memory, model_metrics, feature_importances)
    
    # Save the report to a file
    report_path = os.path.join(app.config['UPLOAD_FOLDER'], 'attrition_report.html')
    with open(report_path, 'w') as f:
        f.write(report_html)
    
    return send_file(report_path, as_attachment=True, download_name='attrition_report.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Create required directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('models', exist_ok=True)
    app.run(debug=True)


@app.route('/api/reset_state', methods=['POST'])
def reset_state():
    # Clear relevant session data
    session_keys_to_clear = [
        'data_preview', 
        'feature_columns', 
        'categorical_columns', 
        'numeric_ranges',
        'accuracy', 
        'data_uploaded',
        'uploaded_filename',
        'clear_form_data'
    ]
    
    for key in session_keys_to_clear:
        if key in session:
            session.pop(key, None)
    
    # Reset global variables
    global df_memory, model_memory, feature_columns
    global categorical_columns, numeric_ranges, scaler
    global label_encoders, model_metrics, feature_importances
    
    df_memory = None
    model_memory = None
    feature_columns = []
    categorical_columns = {}
    numeric_ranges = {}
    scaler = None
    label_encoders = {}
    model_metrics = {}
    feature_importances = {}
    
    return jsonify({"success": True})