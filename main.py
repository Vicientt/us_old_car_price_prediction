from src.car_price import CarPricePredictor

# Dữ liệu mẫu (Đã thêm 'size')
sample_car_raw = {
    "id": 7316814884,
    "url": "https://...",
    "region": "auburn",
    "year": 2015,
    "manufacturer": "toyota",
    "model": "camry",
    "condition": "excellent",
    "cylinders": "4 cylinders",
    "fuel": "gas",
    "odometer": 85000,
    "title_status": "clean",
    "transmission": "automatic",
    
    # Các cột dùng tên gốc (Raw name)
    "drive": "fwd",
    "type": "sedan",
    "paint_color": "white",
    "state": "al",
    
    # THÊM CỘT SIZE (Nếu bạn biết, nếu không code config ở trên sẽ tự điền 'other')
    "size": "mid-size" 
}

if __name__ == "__main__":
    predictor = CarPricePredictor()
    try:
        price = predictor.predict(sample_car_raw)
        print(f"\nVehicle: {sample_car_raw['year']} {sample_car_raw['manufacturer']} {sample_car_raw['model']}")
        print(f"Predicted Price: ${price[0]:,.2f}")
    except Exception as e:
        print(f"An error occurred: {e}")