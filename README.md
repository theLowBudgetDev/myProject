# 🚗 Predictive Maintenance System for Automobiles

This system predicts whether a vehicle is likely to need maintenance soon based on historical, sensor, and usage data.

Developed by **Onah Enrich**, May 2025.

---

## 📊 Key Features

- 🔍 Predict vehicle maintenance status using a trained Random Forest model
- 📈 Visualize risk level with a live gauge meter
- ⚠️ Get actionable maintenance tips based on system analysis
- 🧪 View sensor status compared to normal operating ranges
- 📝 Download a styled PDF report with risk and recommendations
- 📚 View feature importance and explore previous predictions

---

## 🧰 Technologies Used

- Python 3.10+
- Streamlit
- Scikit-learn
- Pandas & NumPy
- Plotly & Seaborn
- FPDF (PDF report generation)
- Kaleido (Plotly image export)

---

## 🚀 How to Run

### Step 1: Clone the Project

    ```bash
    git clone https://github.com/your-username/predictive-maintenance-system.git
    cd predictive-maintenance-system

### Step 2: Install Requirements

    ```bash
    pip install -r requirements.txt

### Step 3: Launch the App

    ```bash
    streamlit run app/app.py

## Project Structure

├── app/
│   ├── app.py                 # Streamlit frontend
│   └── images/logo.png        # Logo for PDF
├── data/
│   ├── vehicle_maintenance_data.csv
│   ├── processed_data.csv
│   └── balanced_data.csv
├── models/
│   ├── rf_model.joblib
│   ├── feature_columns.csv
│   └── normal_ranges.json
├── src/
│   ├── plots/
├   ├── eda.ipynb       # EDA visuals
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
├── requirements.txt
└── README.md

## Acknowledgements

Developed as part of a final year project at Federal University Lafia, Faculty of Computing, Department of Computer Science, 2025.

## Author

Onah Enrich Ugboji
<onahenrich@gmail.com>
