import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import os

# --- PATH CONFIGURATION ---
# This file is located at: project_root/src/car_price/evaluate.py

# 1. Determine the Project Root
# .parent gives 'src/car_price'
# .parent.parent gives 'src'
# .parent.parent.parent gives 'project_root'
current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent.parent

# 2. Model Path
MODEL_PATH = PROJECT_ROOT / 'models' / 'best_xgb_model.pkl'

# 3. Data Path (Updated to data/processed)
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
X_TEST_PATH = DATA_DIR / 'X_test_final.csv'
Y_TEST_PATH = DATA_DIR / 'y_test_final.csv'

def evaluate_model():
    """
    Load the model and test data from data/processed to calculate performance metrics.
    """
    print(f"--- Starting Model Evaluation ---")
    print(f"Project Root detected as: {PROJECT_ROOT}")
    
    # Check if files exist
    if not X_TEST_PATH.exists() or not Y_TEST_PATH.exists():
        print(f"Error: Data files not found at {DATA_DIR}")
        print("Please check if 'X_test_final.csv' and 'y_test_final.csv' are in 'data/processed'.")
        return

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Load Data
    print(f"Loading data from: {DATA_DIR.relative_to(PROJECT_ROOT)} ...")
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)
    
    print(f"Loading model: {MODEL_PATH.name} ...")
    model = joblib.load(MODEL_PATH)
    
    # Predict (Output is in Log Scale)
    print("Running predictions...")
    y_pred_log = model.predict(X_test)
    
    # Flatten y_test to a 1D array
    y_test_log = y_test.values.ravel() 

    # Inverse Transform (Log Scale -> Real Price)
    y_pred_real = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test_log)
    
    # Calculate Metrics
    r2 = r2_score(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    mse = mean_squared_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mse)
    
    # Print Report
    print("\n" + "="*40)
    print(f"MODEL PERFORMANCE REPORT")
    print("="*40)
    print(f"Model used: {MODEL_PATH.name}")
    print("-" * 40)
    print(f"R-squared (R²): {r2:.4f}")
    print(f"MAE (Mean Absolute Error): ${mae:,.2f}")
    print(f"RMSE (Root Mean Sq. Error): ${rmse:,.2f}")
    print("="*40)
    
    # Sample Comparison
    print("\nSample Comparison (First 5 predictions):")
    comparison_df = pd.DataFrame({
        'Actual': y_test_real[:5],
        'Predicted': y_pred_real[:5],
        'Difference': y_pred_real[:5] - y_test_real[:5]
    })
    
    # Format output for currency
    pd.options.display.float_format = '${:,.2f}'.format
    print(comparison_df)

if __name__ == "__main__":
    evaluate_model()