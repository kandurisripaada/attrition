import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

def train_attrition_model(df, model_choice='random_forest', use_grid_search=True, scoring='accuracy'):
    """
    Train a machine learning model for employee attrition prediction with hyperparameter tuning.
    
    Args:
        df (DataFrame): Input DataFrame with employee data.
        model_choice (str): Model type to train ('logistic', 'random_forest', or 'decision_tree').
        use_grid_search (bool): If True, performs hyperparameter tuning using GridSearchCV.
        scoring (str): Scoring metric for model evaluation ('accuracy', 'precision', 'recall', 'f1').
        
    Returns:
        tuple: (best_model, scaler, metrics, feature_importances)
    """
    df = df.copy()
    
    if 'Attrition' in df.columns:
        if df['Attrition'].dtype == object:
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    else:
        raise ValueError("Attrition column not found in dataset")
    
    X = df.drop(columns=['Attrition'], errors='ignore')
    y = df['Attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define model parameters for tuning
    param_grid = {}
    
    if model_choice == 'logistic':
        model = LogisticRegression(max_iter=2000, random_state=42)
        
        # Create compatible parameter combinations for logistic regression
        # Option 1: l1 penalty with liblinear or saga solvers
        l1_params = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l1'],
            'solver': ['liblinear', 'saga']
        }
        
        # Option 2: l2 penalty with any solver (except 'saga' for efficiency)
        l2_params = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear']
        }
        
        # Option 3: elasticnet penalty with saga solver only
        elasticnet_params = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['elasticnet'],
            'solver': ['saga'],
            'l1_ratio': [0.2, 0.5, 0.8]  # Only needed for elasticnet
        }
        
        # Combine into separate parameter grids
        if use_grid_search:
            param_grid = [l1_params, l2_params, elasticnet_params]
        else:
            # For RandomizedSearchCV, we need a different approach
            # Will select the specific parameter grid based on random selection
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty_solver': [
                    ('l1', 'liblinear'),
                    ('l1', 'saga'),
                    ('l2', 'newton-cg'),
                    ('l2', 'lbfgs'),
                    ('l2', 'liblinear'),
                    ('elasticnet', 'saga')
                ],
                'l1_ratio': [0.2, 0.5, 0.8]  # Only used with elasticnet
            }
            # Custom RandomizedSearchCV with parameter dependency handling will be implemented
    
    elif model_choice == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    elif model_choice == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    else:
        raise ValueError(f"Invalid model choice: {model_choice}")
    
    # Perform Hyperparameter Tuning
    best_model = None
    best_params = None
    
    if model_choice == 'logistic' and not use_grid_search:
        # Handle parameter dependencies for RandomizedSearch with logistic regression
        from sklearn.model_selection import KFold
        
        n_iter = 10
        best_score = -1
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for _ in range(n_iter):
            # Randomly select C value
            C = np.random.choice(param_grid['C'])
            
            # Randomly select penalty and solver combination
            penalty, solver = param_grid['penalty_solver'][np.random.randint(0, len(param_grid['penalty_solver']))]
            
            # Set l1_ratio if elasticnet is selected
            l1_ratio = np.random.choice(param_grid['l1_ratio']) if penalty == 'elasticnet' else None
            
            # Configure model
            params = {'C': C, 'penalty': penalty, 'solver': solver}
            if l1_ratio is not None:
                params['l1_ratio'] = l1_ratio
                
            # Create and fit model with these parameters
            model_config = LogisticRegression(random_state=42, max_iter=2000, **params)
            
            # Cross-validation
            scores = []
            for train_idx, valid_idx in cv.split(X_train_scaled):
                X_train_cv, X_valid_cv = X_train_scaled[train_idx], X_train_scaled[valid_idx]
                y_train_cv, y_valid_cv = y_train.iloc[train_idx], y_train.iloc[valid_idx]
                
                model_config.fit(X_train_cv, y_train_cv)
                y_pred_cv = model_config.predict(X_valid_cv)
                
                if scoring == 'accuracy':
                    score = accuracy_score(y_valid_cv, y_pred_cv)
                elif scoring == 'precision':
                    score = precision_score(y_valid_cv, y_pred_cv)
                elif scoring == 'recall':
                    score = recall_score(y_valid_cv, y_pred_cv)
                elif scoring == 'f1':
                    score = f1_score(y_valid_cv, y_pred_cv)
                else:
                    score = accuracy_score(y_valid_cv, y_pred_cv)
                
                scores.append(score)
            
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
        
        # Train final model with best parameters
        best_model = LogisticRegression(random_state=42, max_iter=2000, **best_params)
        best_model.fit(X_train_scaled, y_train)
        
    else:
        # For other models or when using GridSearchCV
        if use_grid_search:
            search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
        else:
            search = RandomizedSearchCV(model, param_grid, cv=5, scoring=scoring, n_iter=10, n_jobs=-1, random_state=42)
        
        search.fit(X_train_scaled, y_train)
        
        best_model = search.best_estimator_
        best_params = search.best_params_
    
    # Make predictions
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'best_params': best_params
    }
    
    # Get feature importances
    feature_importances = {}
    if hasattr(best_model, 'coef_'):
        # For logistic regression
        importance = np.abs(best_model.coef_[0])
        for i, col in enumerate(X.columns):
            feature_importances[col] = float(importance[i])
    elif hasattr(best_model, 'feature_importances_'):
        # For tree-based models
        for i, col in enumerate(X.columns):
            feature_importances[col] = float(best_model.feature_importances_[i])
    
    # Save model
    model_path = f'models/attrition_model.pkl'
    joblib.dump(best_model, model_path)
    
    # Save scaler
    scaler_path = f'models/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    
    print(f"Best Model Trained: {model_choice}")
    print(f"Best Parameters: {best_params}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    return best_model, scaler, metrics, feature_importances