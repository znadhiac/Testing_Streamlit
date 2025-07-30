import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cache model loading to speed up repeated calls
@st.cache_resource
def load_model(path: str = 'final_model_pipeline.pkl'):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Main function to run the Streamlit app
def main():
    # Page configuration
    st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
    
    # Title and description
    st.title("Customer Churn Prediction App")
    st.subheader("Predict customer churn risk using a trained machine learning model")
    st.markdown(
        "This app predicts whether a customer will churn based on behavioral and demographic features."
    )

    # Load model
    model = load_model()
    if model is None:
        st.warning("Model not available. Ensure 'final_model_pipeline.pkl' is in the app directory.")
        return

    # Sidebar for inputs
    st.sidebar.header("Input Customer Features")
    # Input fields
    tenure = st.sidebar.number_input('Tenure (months)', 0, 60, 12)
    city_tier = st.sidebar.selectbox('City Tier', [1, 2, 3])
    warehouse_to_home = st.sidebar.number_input('Warehouse to Home Distance (km)', 0.0, 100.0, 10.0)
    hours_on_app = st.sidebar.slider('Hours Spent on App per Day', 0.0, 24.0, 1.0)
    devices_registered = st.sidebar.number_input('Number of Devices Registered', 1, 10, 2)
    satisfaction = st.sidebar.slider('Satisfaction Score (1 = Low to 5 = High)', 1, 5, 3)
    complaints = st.sidebar.selectbox('Has Complaints?', [0, 1])
    active_user = st.sidebar.selectbox('Is Active User?', [0, 1])
    addresses = st.sidebar.number_input('Number of Addresses', 1, 10, 1)
    hike_pct = st.sidebar.number_input('Order Amount Increase (%)', 0.0, 100.0, 10.0)
    coupons_used = st.sidebar.number_input('Coupons Used', 0, 20, 0)
    orders = st.sidebar.number_input('Order Count', 0, 100, 1)
    days_since_order = st.sidebar.number_input('Days Since Last Order', 0, 365, 10)
    cashback = st.sidebar.number_input('Cashback Amount', 0.0, 1000.0, 0.0)
    recency = st.sidebar.number_input('Recency Ratio', 0.0, 10.0, 1.0)
    unhappy = st.sidebar.selectbox('Unhappy Customer?', [0, 1])
    login_device = st.sidebar.selectbox('Preferred Login Device', ['Mobile Phone', 'Computer'])
    payment_mode = st.sidebar.selectbox('Preferred Payment Mode', ['Debit Card', 'UPI', 'Credit Card', 'Cash on Delivery', 'E-wallet'])
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    order_category = st.sidebar.selectbox('Preferred Order Category', ['Laptop & Accessory', 'Mobile Phone', 'Others', 'Fashion', 'Grocery'])
    marital = st.sidebar.selectbox('Marital Status', ['Single', 'Divorced', 'Married'])
    tenure_group = st.sidebar.selectbox('Tenure Group', ['Early', 'New', 'MidTerm', 'LongTerm'])

    # Assemble input data
    input_dict = {
        'Tenure': tenure,
        'CityTier': city_tier,
        'WarehouseToHome': warehouse_to_home,
        'HourSpendOnApp': hours_on_app,
        'NumberOfDeviceRegistered': devices_registered,
        'SatisfactionScore': satisfaction,
        'NumberOfAddress': addresses,
        'Complain': complaints,
        'OrderAmountHikeFromlastYear': hike_pct,
        'CouponUsed': coupons_used,
        'OrderCount': orders,
        'DaySinceLastOrder': days_since_order,
        'CashbackAmount': cashback,
        'RecencyRatio': recency,
        'IsActiveUser': active_user,
        'UnhappyCustomer': unhappy,
        'PreferredLoginDevice': login_device,
        'PreferredPaymentMode': payment_mode,
        'Gender': gender,
        'PreferedOrderCat': order_category,
        'MaritalStatus': marital,
        'TenureGroup': tenure_group
    }
    input_df = pd.DataFrame([input_dict])

    # Prediction
    if st.sidebar.button('Predict'):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        if prediction == 1:
            st.error(f"Churn: YES (Probability: {probability:.2f})")
        else:
            st.success(f"Churn: NO (Probability: {1 - probability:.2f})")

    # Instructions
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("1. Place 'final_model_pipeline.pkl' in the app directory.")
    st.markdown("2. Install dependencies: `pip install streamlit pandas scikit-learn joblib`.")
    st.markdown("3. Run: `streamlit run app.py`.")

if __name__ == '__main__':
    main()
