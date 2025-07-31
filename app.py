import streamlit as st
import pandas as pd
import pickle

st.title("Churn Prediction App")
st.write("Provide the input features in the sidebar and click 'Predict' to see the predicted churn status.")

@st.cache_resource  # new caching decorator for resources like models
def load_model(pickle_path: str = 'final_model_pipeline.pkl'):
    try:
        with open(pickle_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file '{pickle_path}' not found. Please upload it to the app directory.")
        return None

model = load_model()

if model is not None:
    st.sidebar.header("Input Features")

    tenure = st.sidebar.number_input('Tenure (months)', 0, 60, 12)
    city_tier = st.sidebar.selectbox('City Tier', [1, 2, 3])
    warehouse_to_home = st.sidebar.number_input('Warehouse to Home Distance (km)', 0.0, 100.0, 10.0)
    hour_spend_on_app = st.sidebar.slider('Hours Spent on App per Day', 0.0, 5.0, 1.0)
    num_device_registered = st.sidebar.number_input('Number of Devices Registered', 1, 10, 2)
    satisfaction_score = st.sidebar.slider('Satisfaction Score (1=Low to 5=High)', 1, 5, 3)
    number_of_address = st.sidebar.number_input('Number of Addresses', 1, 10, 1)
    complain = st.sidebar.selectbox('Has Complaints?', [0, 1])
    order_amount_hike = st.sidebar.number_input('Order Amount Hike from Last Year (%)', 0.0, 100.0, 10.0)
    coupon_used = st.sidebar.number_input('Coupons Used', 0, 20, 0)
    order_count = st.sidebar.number_input('Order Count', 0, 50, 1)
    days_since_last_order = st.sidebar.number_input('Days Since Last Order', 0, 100, 10)
    cashback_amount = st.sidebar.number_input('Cashback Amount', 0, 500, 0)
    recency_ratio = st.sidebar.number_input('Recency Ratio', 0.0, 10.0, 1.0)
    is_active_user = st.sidebar.selectbox('Is Active User?', [0, 1])
    unhappy_customer = st.sidebar.selectbox('Is Unhappy Customer?', [0, 1])

    preferred_login_device = st.sidebar.selectbox('Preferred Login Device', ['Mobile Phone', 'Computer'])
    preferred_payment_mode = st.sidebar.selectbox('Preferred Payment Mode', ['Debit Card', 'UPI', 'Credit Card', 'Cash on Delivery', 'E wallet'])
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    preferred_order_cat = st.sidebar.selectbox('Preferred Order Category', ['Laptop & Accessory', 'Mobile Phone', 'Others', 'Fashion', 'Grocery'])
    marital_status = st.sidebar.selectbox('Marital Status', ['Single', 'Divorced', 'Married'])
    tenure_group = st.sidebar.selectbox('Tenure Group', ['Early', 'New', 'MidTerm', 'LongTerm'])

    input_dict = {
        'Tenure': tenure,
        'PreferredLoginDevice': preferred_login_device,
        'CityTier': city_tier,
        'WarehouseToHome': warehouse_to_home,
        'PreferredPaymentMode': preferred_payment_mode,
        'Gender': gender,
        'HourSpendOnApp': hour_spend_on_app,
        'NumberOfDeviceRegistered': num_device_registered,
        'PreferedOrderCat': preferred_order_cat,
        'SatisfactionScore': satisfaction_score,
        'MaritalStatus': marital_status,
        'NumberOfAddress': number_of_address,
        'Complain': complain,
        'OrderAmountHikeFromlastYear': order_amount_hike,
        'CouponUsed': coupon_used,
        'OrderCount': order_count,
        'DaySinceLastOrder': days_since_last_order,
        'CashbackAmount': cashback_amount,
        'RecencyRatio': recency_ratio,
        'IsActiveUser': is_active_user,
        'UnhappyCustomer': unhappy_customer,
        'TenureGroup': tenure_group
    }

    input_df = pd.DataFrame([input_dict])

    if st.sidebar.button('Predict'):
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[:, 1]
        if prediction[0] == 1:
            st.success(f"Churn Prediction: Yes (Probability: {proba[0]:.2f})")
        else:
            st.success(f"Churn Prediction: No (Probability: {1 - proba[0]:.2f})")

else:
    st.warning("Model not loaded. Please upload the 'final_model_pipeline.pkl' file to the app directory.")

st.markdown(
    "---\n"
    "**Instructions:**\n"
    "1. Place `final_model_pipeline.pkl` in the same directory as this script.\n"
    "2. Install dependencies: `pip install streamlit pandas scikit-learn imbalanced-learn`.\n"
    "3. Run the app: `streamlit run app.py`."
)
