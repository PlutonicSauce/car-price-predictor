import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# load everything
@st.cache_resource
def load_models():
    model = joblib.load('car_price_model.pkl')
    encoders = joblib.load('label_encoders.pkl')
    features = joblib.load('feature_cols.pkl')
    metadata = joblib.load('model_metadata.pkl')
    return model, encoders, features, metadata

try:
    model, encoders, feature_cols, metadata = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error("Model files not found. Run train_model.py first")
    # print(f"Error: {e}")  # helpful for debugging

st.title("🚗 Car Price Predictor")
st.write("Enter car details to get a price estimate")

if model_loaded:
    # show some model stats
    with st.expander("Model Info"):
        st.write(f"**Model:** {metadata['best_model']}")
        st.write(f"**R² Score:** {metadata['r2_score']:.4f}")
        st.write(f"**Avg Error:** ${metadata['mae']:,.0f}")
        st.write(f"**Trained:** {metadata['training_year']}")
    
    st.write("---")
    
    # input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Car Details")
        
        make_options = list(encoders['make'].classes_)
        model_options = list(encoders['model'].classes_)
        
        make = st.selectbox("Make", options=sorted(make_options))
        model_name = st.selectbox("Model", options=sorted(model_options))
        year = st.number_input("Year", min_value=1990, max_value=datetime.now().year, 
                               value=2020, step=1)
        mileage = st.number_input("Mileage", min_value=0, max_value=500000, 
                                  value=50000, step=1000)
    
    with col2:
        st.subheader("Additional Info")
        
        fuel_options = list(encoders['fuel_type'].classes_)
        transmission_options = list(encoders['transmission'].classes_)
        
        fuel_type = st.selectbox("Fuel Type", options=sorted(fuel_options))
        transmission = st.selectbox("Transmission", options=sorted(transmission_options))
        
        st.write("")
        st.write("")
        predict_button = st.button("Get Price Estimate", type="primary", use_container_width=True)
    
    if predict_button:
        # setup features
        current_year = datetime.now().year
        training_year = metadata['training_year']
        
        car_age = current_year - year
        age_squared = car_age ** 2
        mileage_per_year = mileage / (car_age + 1) if car_age > 0 else mileage
        high_mileage = 1 if mileage > 100000 else 0
        
        # encode categories
        try:
            make_encoded = encoders['make'].transform([make])[0]
            model_encoded = encoders['model'].transform([model_name])[0]
            fuel_encoded = encoders['fuel_type'].transform([fuel_type])[0]
            trans_encoded = encoders['transmission'].transform([transmission])[0]
        except:
            st.error("Selected option not in training data. Try different values.")
            st.stop()
        
        features_dict = {
            'year': year,
            'mileage': mileage,
            'car_age': car_age,
            'age_squared': age_squared,
            'mileage_per_year': mileage_per_year,
            'high_mileage': high_mileage,
            'make_encoded': make_encoded,
            'model_encoded': model_encoded,
            'fuel_type_encoded': fuel_encoded,
            'transmission_encoded': trans_encoded
        }
        
        input_data = pd.DataFrame([features_dict])
        input_data = input_data[feature_cols]
        
        # get prediction
        prediction = model.predict(input_data)[0]
        
        # adjust for depreciation since training
        years_since_training = current_year - training_year
        if years_since_training > 0:
            depreciation_factor = 0.90 ** years_since_training
            adjusted_prediction = prediction * depreciation_factor
        else:
            adjusted_prediction = prediction
        
        # show results
        st.write("---")
        st.subheader("Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Estimated Price", f"${adjusted_prediction:,.0f}")
        
        with col2:
            lower = adjusted_prediction * 0.90
            upper = adjusted_prediction * 1.10
            st.metric("Range", f"${lower:,.0f} - ${upper:,.0f}")
        
        with col3:
            st.metric("Car Age", f"{car_age} years")
        
        st.info(f"Based on similar {make} {model_name} vehicles. Actual price depends on condition, location, and features.")
        
        # details
        with st.expander("How this was calculated"):
            st.write(f"Year: {year}")
            st.write(f"Mileage: {mileage:,}")
            st.write(f"Age: {car_age} years")
            st.write(f"Miles/year: {mileage_per_year:,.0f}")
            st.write(f"High mileage: {'Yes' if high_mileage else 'No'}")
            if years_since_training > 0:
                st.write(f"\nAdjusted for {years_since_training} year(s) of additional depreciation")

else:
    st.warning("Train the model first: python train_model.py")

st.write("---")
st.caption("Car price predictions using machine learning")
