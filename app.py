import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# -------------------------------------------------------------------
# 1. STYLE & CSS (Phiên bản Gradient Đen 🌑)
# -------------------------------------------------------------------
def apply_custom_styling():
    """
    Áp dụng CSS nội tuyến để tạo phong cách "dark mode" gradient.
    """
    css = """
    <style>
        /* --- Font chữ (vẫn giữ) --- */
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600&display=swap');
        
        html, body, [class*="st-"], .st-emotion-cache-10trblm, .st-emotion-cache-1kyxreq {
            font-family: 'Quicksand', sans-serif;
            color: #f0f0f0; /* Đổi TOÀN BỘ chữ thành màu trắng xám */
        }
        
        /* --- Nền Gradient Đen --- */
        .stApp {
            /* Đây là phần thay đổi chính */
            background-image: linear-gradient(to bottom, #2c3e50, #000000);
            background-attachment: fixed; /* Giữ gradient cố định khi cuộn */
        }
        
        /* --- Tiêu đề chính --- */
        h1 {
            font-family: 'Quicksand', sans-serif;
            font-weight: 600;
            color: #ffffff; /* Chữ trắng */
        }
        
        /* --- Các khối "Card" --- */
        .st-form, .st-container[border="true"] {
            background-color: rgba(26, 26, 26, 0.8); /* Màu xám đen, hơi trong suốt */
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            border: 1px solid #444444; /* Viền xám tối */
        }
        .st-form { border: none !important; }

        /* --- Input và Nút bấm --- */
        /* Đảm bảo chữ label của input cũng màu trắng */
        .st-emotion-cache-1kyxreq, .st-emotion-cache-1n76iwg {
            color: #f0f0f0 !important;
        }

        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div {
            border-radius: 8px;
            border: 1px solid #555555; /* Viền xám */
            background-color: #333333; /* Nền input xám đậm */
            color: #ffffff; /* Chữ gõ vào màu trắng */
        }
        
        /* Nút bấm chính */
        .stButton > button {
            background-color: #007bff; /* Màu xanh dương sáng (nổi bật trên nền đen) */
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background-color: #0056b3;
            transform: scale(1.02);
        }
        
        /* Thông báo thành công */
        .stSuccess {
            background-color: #1a4d2e; /* Nền xanh lá cây đậm */
            border-radius: 8px;
            border: 1px solid #2a7d4e;
            color: #ffffff; /* Chữ trắng */
        }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. TẢI MODEL VÀ PREPROCESSOR
# -------------------------------------------------------------------
@st.cache_resource
def load_model_and_preprocessor():
    try:
        preprocessor_path = os.path.join('models', 'preprocessor.pkl')
        model_path = os.path.join('models', 'best_xgb_model.pkl')
        
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        return preprocessor, model
    except FileNotFoundError as e:
        st.error(f"Error: Model or preprocessor file not found. Make sure the file exists at: {e.filename}")
        st.error("Please ensure 'app.py' is in your root directory, alongside the 'models/' folder.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        return None, None

preprocessor, model = load_model_and_preprocessor()

# -------------------------------------------------------------------
# 3. ĐỊNH NGHĨA CÁC LIST LỰA CHỌN (Options)
# -------------------------------------------------------------------
# (Giữ nguyên các list options của bạn, không cần thay đổi)
manufacturer_options = ['ford', 'chevrolet', 'toyota', 'honda', 'nissan', 'jeep', 'gmc', 'ram', 'dodge', 'bmw', 'mercedes-benz', 'subaru', 'volkswagen', 'hyundai', 'kia', 'lexus', 'audi', 'cadillac', 'acura', 'chrysler', 'lincoln', 'buick', 'infiniti', 'mazda', 'mitsubishi', 'porsche', 'land rover', 'jaguar', 'volvo', 'mini', 'fiat', 'tesla', 'rover', 'alfa-romeo', 'harley-davidson', 'saturn', 'pontiac', 'mercury', 'datsun', 'aston-martin', 'ferrari', 'other']
cylinders_options = ['6 cylinders', '8 cylinders', '4 cylinders', '5 cylinders', '10 cylinders', '3 cylinders', '12 cylinders', 'other']
fuel_options = ['gasoline', 'diesel', 'hybrid', 'electric', 'other']
title_status_options = ['clean', 'rebuilt', 'salvage', 'lien', 'parts only', 'missing']
transmission_options = ['automatic', 'manual', 'other']
type_of_drive_options = ['4wd', 'fwd', 'rwd', 'other']
size_options = ['full-size', 'mid-size', 'compact', 'sub-compact', 'other']
generic_type_options = ['SUV', 'sedan', 'truck', 'pickup', 'coupe', 'hatchback', 'convertible', 'wagon', 'van', 'mini-van', 'offroad', 'bus', 'other']
color_options = ['white', 'black', 'silver', 'blue', 'red', 'grey', 'green', 'brown', 'custom', 'yellow', 'orange', 'purple', 'other']
state_of_listing_options = ['ca', 'tx', 'fl', 'ny', 'pa', 'oh', 'il', 'mi', 'nc', 'ga', 'nj', 'va', 'wa', 'az', 'tn', 'co', 'in', 'mn', 'ma', 'wi', 'md', 'sc', 'mo', 'ct', 'or', 'ky', 'al', 'la', 'ok', 'ut', 'ks', 'nv', 'ia', 'ar', 'ms', 'nm', 'nh', 'ne', 'wv', 'hi', 'me', 'id', 'vt', 'de', 'ak', 'ri', 'mt', 'wy', 'sd', 'nd', 'dc', 'other']

# -------------------------------------------------------------------
# 4. XÂY DỰNG GIAO DIỆN WEB (UI)
# -------------------------------------------------------------------

# Cấu hình trang (Page Config)
st.set_page_config(
    page_title="Old Car Price Predictor",
    page_icon="🤖",
    layout="wide"
)

# Áp dụng CSS
apply_custom_styling()

# Tiêu đề
st.title('🤖 Old Car Price Predictor 🚗')
st.write("Fill in the car's details below to get a price prediction.")

# Chỉ chạy app nếu model được tải thành công
if preprocessor is not None and model is not None:
    
    # Sử dụng form để gom nhóm input
    with st.form(key='input_form'):
        
        # Tạo 2 cột
        col1, col2 = st.columns(2)

        # --- CỘT 1 ---
        with col1:
            with st.container(border=True):
                st.subheader("Technical Specs ⚙️")
                year_to_now = st.number_input('Vehicle Age (Years)', min_value=0, max_value=80, value=5, step=1,
                                             help="Current year minus the model year. (e.g., 2024 - 2019 = 5)")
                odometer = st.number_input('Odometer (miles)', min_value=0, max_value=1000000, value=50000, step=1000)
                cylinders = st.selectbox('Cylinders', options=cylinders_options, index=0)
                fuel = st.selectbox('Fuel Type', options=fuel_options, index=0)
                transmission = st.selectbox('Transmission', options=transmission_options, index=0)
                type_of_drive = st.selectbox('Drive Type', options=type_of_drive_options, index=0)

        # --- CỘT 2 ---
        with col2:
            with st.container(border=True):
                st.subheader("Identification 🎨")
                manufacturer = st.selectbox('Manufacturer', options=manufacturer_options, index=0)
                model_input = st.text_input('Vehicle Model (e.g., f-150, camry)', value='other',
                                            help="Enter the model name. If it's rare or unknown, leave as 'other'.")
                size = st.selectbox('Size', options=size_options, index=0)
                generic_type = st.selectbox('Vehicle Type', options=generic_type_options, index=0)
                color = st.selectbox('Color', options=color_options, index=0)
                title_status = st.selectbox('Title Status', options=title_status_options, index=0)

        # --- Input cuối cùng (trải dài) ---
        with st.container(border=True):
            state_of_listing = st.selectbox('State of Listing (US)', options=state_of_listing_options, index=0)
        
        st.markdown("<br>", unsafe_allow_html=True) # Thêm một chút khoảng cách

        # Nút submit cho form
        submit_button = st.form_submit_button(
            label='Predict Price', 
            use_container_width=True
        )

    # -------------------------------------------------------------------
    # 5. XỬ LÝ PREDICTION (Chỉ chạy khi nhấn nút)
    # -------------------------------------------------------------------
    if submit_button:
        try:
            # 1. Tạo DataFrame từ input
            input_data = pd.DataFrame({
                'manufacturer': [manufacturer],
                'model': [model_input.lower()],
                'cylinders': [cylinders],
                'fuel': [fuel],
                'odometer': [odometer],
                'title_status': [title_status],
                'transmission': [transmission],
                'type_of_drive': [type_of_drive],
                'size': [size],
                'generic_type': [generic_type],
                'color': [color],
                'state_of_listing': [state_of_listing],
                'year_to_now': [year_to_now]
            })

            # 2. Biến đổi dữ liệu
            transformed_data = preprocessor.transform(input_data)
            
            # 3. Dự đoán
            price_log = model.predict(transformed_data)
            
            # 4. Chuyển đổi ngược
            price_final = np.expm1(price_log)
            
            # 5. Hiển thị kết quả
            st.success(f'The predicted price is: ${price_final[0]:,.2f}')

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please check the input values and ensure the preprocessor was trained correctly.")
else:
    st.error("Models are not loaded. The app cannot run.")