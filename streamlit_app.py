import joblib
import gzip
import requests
import io

# Replace this with your actual direct download URL
url = "https://drive.google.com/file/d/1oISJqpZMRAW-FF-E5yHDYKbMkTOtB2TL/view?usp=sharing"

response = requests.get(url)
compressed_model = io.BytesIO(response.content)

with gzip.open(compressed_model, 'rb') as f:
    model = joblib.load(f)


st.title("🔋 Energy Prediction & EMS Dashboard")
st.markdown("This dashboard predicts energy consumption and applies a basic EMS strategy.")

# --- Upload CSV file ---
uploaded_file = st.file_uploader("📁 Upload your dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")
    st.write("Preview:", df.head())

    # --- Make predictions ---
    features = df.drop(columns=["target", "anomaly_type"], errors='ignore')
    predictions = model.predict(features)
    df['Predicted_Energy'] = predictions

    # --- Simple EMS Strategy (Example) ---
    def ems_strategy(row):
        if row["Predicted_Energy"] > 0.5:
            return "Use solar + grid"
        else:
            return "Use battery"
    
    df["EMS_Action"] = df.apply(ems_strategy, axis=1)

    # --- Visualizations ---
    st.subheader("📊 Prediction vs Time")
    st.line_chart(df["Predicted_Energy"])

    st.subheader("⚡ EMS Actions Summary")
    st.write(df["EMS_Action"].value_counts())

    st.subheader("📥 Download Results")
    st.download_button("Download CSV", df.to_csv(index=False), file_name="ems_results.csv")

