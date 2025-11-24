import joblib
import pandas as pd
import numpy as np
import scipy.sparse
from src.car_price import config, cleaning

class CarPricePredictor:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self._load_models()

    def _load_models(self):
        if not config.PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor not found at {config.PREPROCESSOR_PATH}")
        
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}")

        print(f"Loading model from {config.MODEL_PATH}...")
        self.preprocessor = joblib.load(config.PREPROCESSOR_PATH)
        self.model = joblib.load(config.MODEL_PATH)
        print("Models loaded successfully!")

    def predict(self, data: dict | pd.DataFrame) -> np.ndarray:
        # 1. Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input must be a dictionary or pandas DataFrame")

        # 2. Manual Cleaning
        df_clean = cleaning.clean_data(df)

        # 3. Preprocessing
        try:
            X_transformed = self.preprocessor.transform(df_clean)
            
            if scipy.sparse.issparse(X_transformed):
                X_transformed = X_transformed.toarray()
                
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            print("Current columns:", df_clean.columns)
            raise e

        # 4. Prediction
        y_pred_log = self.model.predict(X_transformed)

        # 5. Inverse Transformation
        y_pred_real = np.expm1(y_pred_log)

        return y_pred_real