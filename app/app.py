import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import os
from fpdf import FPDF
import tempfile
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import plotly.graph_objects as go


# Load Model and Data
model = joblib.load("models/rf_model.joblib")
data = pd.read_csv("data/processed_data.csv")
scaler = StandardScaler()
scaler.fit(data.drop(columns=["Need_Maintenance"]))
# Load feature columns used during model training
feature_order = pd.read_csv("models/feature_columns.csv", header=None)[0].tolist()

# Load normal ranges
with open("models/normal_ranges.json") as f:
    normal_ranges = json.load(f)

# Page Configuration
st.set_page_config(page_title="Predictive Maintenance System", layout="wide")

# Title and Description
st.title("Predictive Maintenance System")
st.markdown("Predict vehicle maintenance needs with our advanced Random Forest model. Input your vehicle details below to get started!")

# Sidebar for Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Predict", "Feature Importance", "Prediction History"])

# Initialize Session State for Prediction History
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Prediction Page
if page == "Predict":
    st.header("Predict Maintenance Needs")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            vehicle_model = st.selectbox("Vehicle Model", ["Truck", "Van", "Bus", "Motorcycle", "SUV", "Car"])
            mileage = st.slider("Mileage", 0, 200000, 50000)
            maintenance_history = st.selectbox("Maintenance History", ["Poor", "Average", "Good"])
            reported_issues = st.slider("Reported Issues", 0, 10, 0)
            vehicle_age = st.slider("Vehicle Age (Years)", 0, 20, 5)
            fuel_type = st.selectbox("Fuel Type", ["Electric", "Petrol", "Diesel"])
            transmission_type = st.selectbox("Transmission Type", ["Automatic", "Manual"])
            engine_size = st.slider("Engine Size (cc)", 500, 5000, 2000)
            odometer_reading = st.slider("Odometer Reading", 0, 200000, 30000)
        with col2:
            last_service_date = st.date_input("Last Service Date", value=datetime(2024, 1, 1))
            warranty_expiry_date = st.date_input("Warranty Expiry Date", value=datetime(2025, 6, 1))
            owner_type = st.selectbox("Owner Type", ["First", "Second", "Third"])
            insurance_premium = st.slider("Insurance Premium", 5000, 50000, 20000)
            service_history = st.slider("Service History (Count)", 0, 20, 5)
            accident_history = st.slider("Accident History", 0, 10, 0)
            fuel_efficiency = st.slider("Fuel Efficiency (km/l)", 5.0, 25.0, 15.0, step=0.1)
            tire_condition = st.selectbox("Tire Condition", ["New", "Good", "Worn Out"])
            brake_condition = st.selectbox("Brake Condition", ["New", "Good", "Worn Out"])
            battery_status = st.selectbox("Battery Status", ["Weak", "Good", "New"])
        
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare Input Data
        from datetime import datetime, date

        input_data = {
            "Mileage": mileage,
            "Reported_Issues": reported_issues,
            "Vehicle_Age": vehicle_age,
            "Engine_Size": engine_size,
            "Odometer_Reading": odometer_reading,
            "Insurance_Premium": insurance_premium,
            "Service_History": service_history,
            "Accident_History": accident_history,
            "Fuel_Efficiency": int(fuel_efficiency),
            "Maintenance_History": {"Poor": 0, "Average": 1, "Good": 2}[maintenance_history],
            "Tire_Condition": {"New": 0, "Good": 1, "Worn Out": 2}[tire_condition],
            "Brake_Condition": {"New": 0, "Good": 1, "Worn Out": 2}[brake_condition],
            "Battery_Status": {"Weak": 0, "Good": 1, "New": 2}[battery_status],
            "Days_Since_Last_Service": (datetime(2025, 5, 28) - datetime.combine(last_service_date, datetime.min.time())).days,
            "Warranty_Expired": 1 if datetime.combine(warranty_expiry_date, datetime.min.time()) < datetime(2025, 5, 28) else 0
        }

        for col in ["Vehicle_Model", "Fuel_Type", "Transmission_Type", "Owner_Type"]:
            categories = {
                "Vehicle_Model": ["Truck", "Van", "Bus", "Motorcycle", "SUV", "Car"],
                "Fuel_Type": ["Electric", "Petrol", "Diesel"],
                "Transmission_Type": ["Automatic", "Manual"],
                "Owner_Type": ["First", "Second", "Third"]
            }
            for cat in categories[col]:
                input_data[f"{col}_{cat}"] = 1 if locals()[col.lower()] == cat else 0

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=feature_order)
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]

        # Display Prediction Result
        if prediction == 1:
            st.error(f"🚨 Maintenance Needed! (Confidence: {prediction_proba:.2%})")
        else:
            st.success(f"✅ No Maintenance Needed. (Confidence: {1 - prediction_proba:.2%})")

        col1, col2 = st.columns([1, 2])

        # Semi-circle gauge chart for prediction probability
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prediction_proba * 100, 2),
            number={'suffix': "%"},
            title={'text': "Maintenance Risk Level"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': "darkred" if prediction_proba > 0.7 else "orange" if prediction_proba > 0.4 else "green"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': 'lightgreen'},
                    {'range': [40, 70], 'color': 'gold'},
                    {'range': [70, 100], 'color': 'lightcoral'}
                ],
            },
            domain={'x': [0, 1], 'y': [0, 1]}
        ))

        gauge_fig.update_layout(
            height=400,
            margin=dict(t=50, b=0, l=0, r=0),
        )

        with col1:
            st.plotly_chart(gauge_fig, use_container_width=True)


        # Compare to normal ranges
        comparison_results = []
        for feature in input_df.columns:
            value = input_df[feature].values[0]
            range_info = normal_ranges.get(feature)
            if range_info:
                lower = range_info["lower"]
                upper = range_info["upper"]
                if value < lower:
                    status = "LOW"
                elif value > upper:
                    status = "HIGH"
                else:
                    status = "Normal"
                comparison_results.append({"Feature": feature, "Value": value, "Normal Min": lower, "Normal Max": upper, "Status": status})

        # Display chart
        comparison_df = pd.DataFrame(comparison_results)
        fig = px.bar(comparison_df, x="Feature", y="Value", color="Status",
                     title="Input Feature Values Compared to Normal Range",
                     labels={"Value": "Current Value"},
                     color_discrete_map={"LOW": "blue", "HIGH": "red", "Normal": "green"})
        
        with col2:
            st.plotly_chart(fig, use_container_width=True)

        # Display table
        st.subheader("🧪 Sensor Health Status Table")
        st.dataframe(comparison_df.style.applymap(
            lambda x: "color: red;" if isinstance(x, str) and "HIGH" in x else
                      "color: blue;" if isinstance(x, str) and "LOW" in x else
                      "color: green;" if isinstance(x, str) and "Normal" in x else "",
            subset=["Status"]
        ))


        # Maintenance Tips
        if prediction == 1:
            st.subheader("🛠️ Recommended Maintenance Actions")
            if input_data["Battery_Status"] == 0:
                st.write("- Replace the battery (Weak status detected).")
            if input_data["Tire_Condition"] == 2:
                st.write("- Replace tires (Worn Out condition).")
            if input_data["Brake_Condition"] == 2:
                st.write("- Inspect and replace brake pads (Worn Out condition).")

        
        #********************
           #PDF GENERATION
        #********************
        def safe_text(text):
            return str(text).encode('latin-1', errors='ignore').decode('latin-1')
        # Save gauge chart image
        gauge_img_path = "app/images/gauge_chart.png"
        try:
            gauge_fig.write_image(gauge_img_path)
        except Exception as e:
            st.warning(f"Gauge chart image could not be saved: {e}")
            gauge_img_path = None

        # Custom styled PDF class
        class StyledPDF(FPDF):

            def header(self):
                try:
                    self.image("app/images/logo.png", x=10, y=8, w=35)
                except:
                    pass
                self.set_font("Helvetica", "B", 16)
                self.cell(0, 10, safe_text("Maintenance Summary Report"), ln=True, align="C")
                self.set_draw_color(0, 0, 0)
                self.set_line_width(0.5)
                self.line(10, 25, 200, 25)
                self.ln(12)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 9)
                self.set_text_color(128)
                self.cell(0, 10, safe_text("(c) 2025 | Predictive Maintenance System by Onah Enrich"), align="C")

            def section_title(self, title):
                self.set_font("Helvetica", "B", 13)
                self.set_text_color(0, 51, 102)
                self.cell(0, 10, title, ln=True)
                self.set_text_color(0, 0, 0)

            def line_text(self, label, value):
                self.set_font("Helvetica", "B", 11)
                self.cell(50, 8, f"{label}:", ln=0)
                self.set_font("Helvetica", "", 11)
                self.multi_cell(0, 8, safe_text(str(value)))


        # Create the styled PDF
        pdf = StyledPDF()
        pdf.add_page()

        # Title section
        pdf.section_title("Prediction Summary")
        pdf.line_text("Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        pdf.line_text("Prediction", "🛠️ Maintenance Needed" if prediction == 1 else "✅ No Maintenance Needed")
        pdf.line_text("Confidence", f"{round(prediction_proba * 100, 2)}%")

        # Risk level
        risk_level = "Low"
        if prediction_proba > 0.7:
            risk_level = "High"
        elif prediction_proba > 0.4:
            risk_level = "Moderate"
        pdf.line_text("Risk Level", risk_level)

        # Add gauge chart image
        if gauge_img_path and os.path.exists(gauge_img_path):
            pdf.ln(5)
            pdf.section_title("Risk Meter")
            pdf.image(gauge_img_path, x=30, w=150)
            pdf.ln(5)

        # Key Sensor Abnormalities
        pdf.section_title("Sensor Alerts")
        abnormal_df = comparison_df[comparison_df["Status"] != "Normal"]
        pdf.set_font("Helvetica", "", 11)
        if not abnormal_df.empty:
            for _, row in abnormal_df.head(3).iterrows():
                status_icon = "🔴" if row["Status"] == "HIGH" else "🔵"
                pdf.multi_cell(0, 8, safe_text(f"- {row['Feature']}: {row['Value']} ({row['Status']}) | Normal: {row['Normal Min']}-{row['Normal Max']}"))
        else:
            pdf.cell(0, 8, safe_text("✅ All critical sensors are within normal range."), ln=True)

        # Maintenance Tips
        pdf.section_title("Recommendations")
        tips_added = False
        if prediction == 1:
            if input_data["Battery_Status"] == 0:
                pdf.cell(0, 8, safe_text("- Replace weak battery."), ln=True)
                tips_added = True
            if input_data["Tire_Condition"] == 2:
                pdf.cell(0, 8, safe_text("- Replace worn-out tires."), ln=True)
                tips_added = True
            if input_data["Brake_Condition"] == 2:
                pdf.cell(0, 8, safe_text("- Service brake system (worn pads)."), ln=True)
                tips_added = True
            if not tips_added:
                pdf.cell(0, 8, safe_text("- Full inspection advised based on system risk score."), ln=True)
        else:
            pdf.cell(0, 8, safe_text("✅ No immediate maintenance action required."), ln=True)

        # Save styled PDF
        pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(pdf_file.name)

        # Download button
        with open(pdf_file.name, "rb") as f:
            st.download_button(
                label="Download Report",
                data=f,
                file_name="maintenance_summary.pdf",
                mime="application/pdf"
            )


        # Save Prediction History
        st.session_state.prediction_history.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Vehicle_Model": vehicle_model,
            "Prediction": "Yes" if prediction == 1 else "No",
            "Confidence": prediction_proba
        })

# Feature Importance Page
elif page == "Feature Importance":
    st.header("Feature Importance")
    importance = model.feature_importances_
    features = data.drop(columns=["Need_Maintenance"]).columns
    importance_df = pd.DataFrame({"Feature": features, "Importance": importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    
    fig = px.bar(importance_df, x="Importance", y="Feature", title="Feature Importance in Random Forest Model")
    st.plotly_chart(fig)

# Prediction History Page
elif page == "Prediction History":
    st.header("Prediction History")
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df)
        csv = history_df.to_csv(index=False)
        st.download_button("Download History", csv, "prediction_history.csv", "text/csv")
    else:
        st.info("No predictions made yet. Go to the Predict page to make a prediction!")

# Footer
st.markdown("---")
st.markdown("Developed by Onah Enrich | Powered by Streamlit | May 2025")
