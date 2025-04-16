import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.f1_lap_predictor import predict_japanese_gp, fetch_recent_data, train_and_evaluate
import fastf1
import pandas as pd

# Initialize cache
fastf1.Cache.enable_cache('../cache')

# App layout
st.set_page_config(page_title="F1 Qualifying Predictor", layout="wide")
st.title("🏎️ F1 2025 Qualifying Predictor")

# Load and train model once
@st.cache_resource
def load_model():
    data = fetch_recent_data()
    if data:
        combined_df = pd.concat(data, ignore_index=True)
        valid_data = combined_df.dropna(subset=['Q1_sec', 'Q2_sec', 'Q3_sec'], how='all')
        return train_and_evaluate(valid_data)
    return None

# Sidebar controls
with st.sidebar:
    st.header("Model Controls")
    year = st.selectbox("Season", [2025, 2024])
    track = st.selectbox("Circuit", ["Suzuka", "Silverstone", "Monza"])
    predict_button = st.button("Run Prediction")

# Main display area
model = load_model()

if not model:
    st.error("Failed to load prediction model. Check data connections.")
    st.stop()

if predict_button:
    with st.spinner("Fetching latest data and running predictions..."):
        try:
            # Get fresh data
            current_data = fetch_recent_data()
            latest_df = pd.concat(current_data, ignore_index=True) if current_data else None
            
            if latest_df is None:
                st.error("Failed to fetch recent race data")
                st.stop()

            # Run prediction with actual model
            predictions = predict_japanese_gp(model, latest_df)
            
            # Display results
            st.subheader(f"{track} {year} Predicted Qualifying Times")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(
                    predictions[['Driver', 'Team', 'Predicted_Q3']],
                    column_config={
                        "Predicted_Q3": st.column_config.NumberColumn(
                            "Q3 Time (s)", format="%.3f"
                        )
                    },
                    hide_index=True
                )
            
            with col2:
                st.bar_chart(
                    predictions.set_index('Driver')['Predicted_Q3'],
                    color="#FF4B4B"
                )
            
            # Show model metrics
            st.subheader("Model Performance")
            st.write(f"Mean Absolute Error: {model['mae']:.3f} seconds")
            st.write(f"R² Score: {model['r2']:.3f}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
