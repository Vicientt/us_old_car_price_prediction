# 🚗 Used Car Price Prediction

## 📖 Overview

This project is an end-to-end Machine Learning solution designed to estimate the market value of used cars based on specifications such as manufacturer, model, year, mileage, and condition.

Originally developed as a research project using Craigslist data, it has been refactored into a **production-ready architecture**. The system features a robust data pipeline, an optimized **XGBoost** inference engine, and a modern **Streamlit** dashboard for real-time user interaction.

## ✨ Key Features

* **Production-Grade Pipeline:** Logic is decoupled into modular components (`src/car_price`) for cleaning, preprocessing, and inference.
* **Advanced Modeling:** Utilizes **XGBoost Regressor** (tuned via RandomizedSearchCV) for high-accuracy predictions, outperforming Ridge Regression and Random Forest benchmarks.
* **Robust Feature Engineering:**
    * Automatic handling of missing values and categorical encoding.
    * **Time-Skew Correction:** Adjusted vehicle age calculation to match the dataset's collection year (2021) rather than the current real-time year.
    * **Sparse/Dense Matrix Handling:** Solved compatibility issues between training data and inference inputs.
* **Modern UI:** A sleek, dark-mode Streamlit dashboard allowing users to input vehicle details and get instant valuations.
* **Fast Dependency Management:** Powered by **[uv](https://github.com/astral-sh/uv)** for ultra-fast package installation and resolution.

## 📂 Project Structure

```text
car-price-prediction/
├── app/
│   └── streamlit_app.py    # The Web Interface (Frontend)
├── data/
│   └── processed/          # Processed datasets (train/test split)
├── models/
│   ├── preprocessor.pkl    # Data transformation pipeline
│   ├── ridge_model.pkl     # Baseline model
│   └── best_xgb_model.pkl  # Production model (XGBoost)
├── notebooks/              # Research & Experiments (EDA, Model Selection)
├── src/
│   └── car_price/          # Core Logic Package
│       ├── cleaning.py     # Data cleaning & Feature engineering logic
│       ├── config.py       # Configuration & Path management
│       ├── inference.py    # Prediction engine class
│       └── evaluate.py     # Performance evaluation script
├── main.py                 # CLI Entry point for quick testing
├── pyproject.toml          # Project configuration & Dependencies
├── uv.lock                 # Lock file for reproducible builds
└── README.md               # Project Documentation
```

## 🚀 Installation & Setup

This project uses **uv** for dependency management.

1.  **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/car-price-prediction.git
    cd car-price-prediction
    ```

2.  **Install dependencies**

    ```bash
    uv sync
    ```

## 💻 Usage

### 1. Run the Web App (Recommended)

```bash
uv run streamlit run app/streamlit_app.py
```

### 2. Run CLI Prediction

```bash
uv run main.py
```

### 3. Evaluate Model Performance

```bash
uv run src/car_price/evaluate.py
```

## 🛠️ Tech Stack

- **Language:** Python 3.12+
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost
- **Web Framework:** Streamlit
- **Package Manager:** uv

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
