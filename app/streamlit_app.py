import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import the Predictor Class from your source code
from src.car_price import CarPricePredictor

# --- 2. PAGE & STYLE CONFIGURATION ---
st.set_page_config(
    page_title="Car Price Prediction", 
    page_icon="🚗", 
    layout="wide" # Use wide layout to support the 2-column design better
)

def apply_custom_styling():
    """
    Apply the "Dark Mode Gradient" CSS design.
    """
    css = """
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600&display=swap');
        
        /* Apply Font & Text Color */
        html, body, [class*="st-"], .st-emotion-cache-10trblm, .st-emotion-cache-1kyxreq {
            font-family: 'Quicksand', sans-serif;
            color: #f0f0f0; 
        }
        
        /* --- Background Gradient --- */
        .stApp {
            background-image: linear-gradient(to bottom, #2c3e50, #000000);
            background-attachment: fixed;
        }
        
        /* --- Header Title --- */
        h1 {
            color: #ffffff;
            text-align: center;
            text-shadow: 0px 0px 10px rgba(255, 255, 255, 0.3);
        }
        
        /* --- Container/Card Styling --- */
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            background-color: rgba(255, 255, 255, 0.05); /* Glass effect */
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 15px;
        }

        /* --- Input Fields Styling --- */
        .stTextInput, .stNumberInput, .stSelectbox {
            color: #ffffff;
        }
        
        /* --- Submit Button Styling --- */
        .stButton > button {
            background: linear-gradient(90deg, #ff8a00, #e52e71);
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 25px;
            padding: 12px 25px;
            width: 100%;
            margin-top: 10px;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(229, 46, 113, 0.6);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply the styles
apply_custom_styling()

# --- 3. LOAD MODEL FROM SRC ---
# Use cache so we don't reload the model on every interaction
@st.cache_resource
def get_predictor():
    return CarPricePredictor()

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 4. UI LAYOUT (MATCHING YOUR ORIGINAL DESIGN) ---

st.title("🤖 Car Price Predictor 🚗")
st.markdown("<p style='text-align: center; color: #bdc3c7;'>XGBoost Powered Engine</p>", unsafe_allow_html=True)
st.write("---")

# --- Define Options Lists (Taken from your original file) ---
manufacturer_opts = ['ford', 'chevrolet', 'toyota', 'honda', 'nissan', 'jeep', 'gmc', 'ram', 'dodge', 'bmw', 'mercedes-benz', 'subaru', 'volkswagen', 'hyundai', 'kia', 'lexus', 'audi', 'cadillac', 'acura', 'chrysler', 'lincoln', 'buick', 'infiniti', 'mazda', 'mitsubishi', 'porsche', 'land rover', 'jaguar', 'volvo', 'mini', 'fiat', 'tesla', 'rover', 'alfa-romeo', 'harley-davidson', 'saturn', 'pontiac', 'mercury', 'datsun', 'aston-martin', 'ferrari', 'other']
cylinders_opts = ['6 cylinders', '8 cylinders', '4 cylinders', '5 cylinders', '10 cylinders', '3 cylinders', '12 cylinders', 'other']
fuel_opts = ['gas', 'diesel', 'hybrid', 'electric', 'other'] 
trans_opts = ['automatic', 'manual', 'other']
drive_opts = ['4wd', 'fwd', 'rwd', 'other']
size_opts = ['full-size', 'mid-size', 'compact', 'sub-compact', 'other']
type_opts = ['SUV', 'sedan', 'truck', 'pickup', 'coupe', 'hatchback', 'convertible', 'wagon', 'van', 'mini-van', 'offroad', 'bus', 'other']
color_opts = ['white', 'black', 'silver', 'blue', 'red', 'grey', 'green', 'brown', 'custom', 'yellow', 'orange', 'purple', 'other']
title_opts = ['clean', 'rebuilt', 'salvage', 'lien', 'parts only', 'missing']

# --- FORM INPUT ---
with st.form(key='prediction_form'):
    col1, col2 = st.columns(2)

    # === COLUMN 1: TECHNICAL SPECS ===
    with col1:
        with st.container(border=True):
            st.subheader("Technical Specs ⚙️")
            
            # Logic: User inputs Age -> We calculate Year
            # The model was trained on 2021 data. If Age = 5, the car is from 2016.
            vehicle_age = st.number_input('Vehicle Age (Years)', min_value=0, max_value=80, value=5, 
                                        help="Example: If the car was made in 2016, the age is 5 (relative to 2021 data).")
            
            odometer = st.number_input('Odometer (miles)', min_value=0, max_value=1000000, value=50000, step=1000)
            cylinders = st.selectbox('Cylinders', options=cylinders_opts)
            fuel = st.selectbox('Fuel Type', options=fuel_opts)
            transmission = st.selectbox('Transmission', options=trans_opts)
            type_of_drive = st.selectbox('Drive Type', options=drive_opts)

    # === COLUMN 2: IDENTIFICATION ===
    with col2:
        with st.container(border=True):
            st.subheader("Identification 🎨")
            manufacturer = st.selectbox('Manufacturer', options=manufacturer_opts)
            model_input = st.text_input('Vehicle Model (e.g. Camry)', value='other')
            size = st.selectbox('Size', options=size_opts)
            generic_type = st.selectbox('Vehicle Type', options=type_opts)
            color = st.selectbox('Color', options=color_opts)
            title_status = st.selectbox('Title Status', options=title_opts)

    # === BOTTOM INPUT ===
    with st.container(border=True):
        state_of_listing = st.text_input('State of Listing (Code, e.g. ca, ny)', value='ca')

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label='PREDICT PRICE')

# --- 5. PREDICTION LOGIC ---
if submit_button:
    # Logic Calculation:
    # XGBoost needs the "Manufactured Year" to compare against the 2021 dataset.
    calculated_year = 2021 - vehicle_age
    
    # Create Dictionary with RAW NAMES 
    # (src/car_price/cleaning.py will handle mapping these raw names to model names)
    input_data = {
        "year": calculated_year,
        "odometer": odometer,
        "cylinders": cylinders,
        "fuel": fuel,
        "transmission": transmission,
        "drive": type_of_drive,         # Raw input name
        
        "manufacturer": manufacturer,
        "model": model_input,
        "size": size,
        "type": generic_type,           # Raw input name
        "paint_color": color,           # Raw input name
        "title_status": title_status,
        "state": state_of_listing       # Raw input name
    }

    with st.spinner("Analyzing market data..."):
        try:
            # Call the predict method from your source code
            price_final = predictor.predict(input_data)
            
            # Display Results
            st.markdown("---")
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: rgba(255,255,255,0.05); border-radius: 15px; border: 1px solid rgba(255,255,255,0.2);">
                <h3 style="margin:0; color: #ecf0f1; font-weight: 400;">Estimated Market Value</h3>
                <h1 style="margin:15px; color: #2ecc71; font-size: 3.8em; text-shadow: 0 0 15px rgba(46, 204, 113, 0.4);">${price_final[0]:,.2f}</h1>
                <p style="color: #bdc3c7;">For a {vehicle_age}-year-old {manufacturer.title()} {model_input.title()}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info("Please check the terminal logs for more details.")