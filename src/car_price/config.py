import os
from pathlib import Path

# --- PATH CONFIGURATION ---
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
MODEL_DIR = PROJECT_ROOT / 'models'

PREPROCESSOR_PATH = MODEL_DIR / 'preprocessor.pkl'
MODEL_PATH = MODEL_DIR / 'best_xgb_model.pkl'

# --- COLUMN MAPPING (Raw Input -> Model Input) ---
# This translates the keys from your scraper/raw data to what the model expects
COLUMN_MAPPING = {
    'paint_color': 'color',        # Raw: paint_color -> Model: color
    'type': 'generic_type',        # Raw: type -> Model: generic_type
    'drive': 'type_of_drive',      # Raw: drive -> Model: type_of_drive
    'state': 'state_of_listing',   # Raw: state -> Model: state_of_listing
    # Add more here if your raw data changes
}

# --- CLEANING CONFIGURATION ---
# Columns to drop (based on your EDA notebook)
DROP_COLUMNS = [
    'id', 'url', 'region_url', 'image_url', 'description', 
    'posting_date', 'lat', 'long', 'county', 'VIN', 'region',
    'condition' # Often has too many nulls, safe to drop if not used
]

# Categorical columns to fill with 'other'
# NOTE: Use the NEW names (Model Input names) here
CAT_COLS_FILL_OTHER = [
    'fuel', 'title_status', 'transmission', 'type_of_drive', 
    'color', 'manufacturer', 'model', 'generic_type', 'state_of_listing'
]

# Numerical columns
NUM_COLS = ['odometer', 'year']