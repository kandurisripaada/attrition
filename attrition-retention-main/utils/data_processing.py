import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def process_data(df: pd.DataFrame, existing_encoders=None) -> tuple:
    """
    Perform basic data processing: handle null values, drop duplicates, and encode categorical variables.
    
    Args:
    df (DataFrame): Input DataFrame to process
    existing_encoders (dict, optional): Existing label encoders to use
    
    Returns:
    tuple: (processed_df, label_encoders)
    """
    df = df.copy()
    
    # Handle missing values with simple approaches
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Use median for numeric columns
                df[col] = df[col].fillna(df[col].median())
            else:
                # For categorical, use mode (most frequent value)
                df[col] = df[col].fillna(df[col].mode()[0])
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Encode categorical variables
    label_encoders = existing_encoders if existing_encoders else {}
    
    for col in df.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            # Use existing encoder
            try:
                df[col] = label_encoders[col].transform(df[col])
            except:
                # If value not in encoder classes, assign -1
                unknown_mask = ~df[col].isin(label_encoders[col].classes_)
                known_mask = ~unknown_mask
                if known_mask.any():
                    df.loc[known_mask, col] = label_encoders[col].transform(df.loc[known_mask, col])
                if unknown_mask.any():
                    df.loc[unknown_mask, col] = -1
        else:
            # Create new encoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    return df, label_encoders