import streamlit as st
import pandas as pd
import pickle

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# --- HEADER ---
st.title("Customer Churn Prediction App")
st.markdown("""
This application estimates the likelihood that a customer will **churn**, meaning they may stop using the platform or making purchases.  
Enter customer details in the sidebar and click **Predict** to assess churn risk.
""")

# --- LOAD MODEL ---
@st.cache_resource
def load_model(pickle_path='final_model.pkl'):
    try:
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please contact the administrator.")
        return None

model = load_model()

# --- USER INPUT ---
if model is not None:
    st.sidebar.header("CUSTOMER DETAILS")

    with st.sidebar:
        st.subheader("Engagement & Usage")
        tenure = st.number_input('Tenure (months)', 0, 60, 12)
        hour_spend_on_app = st.slider('Hours on App/Week', 0.0, 5.0, 1.0)
        num_device_registered = st.number_input('Devices Registered', 1, 10, 2)
        order_count = st.number_input('Order Count', 0, 50, 1)
        days_since_last_order = st.number_input('Days Since Last Order', 0, 100, 10)
        cashback_amount = st.number_input('Cashback Amount', 0, 500, 0)
        recency_ratio = st.number_input('Recency Ratio', 0.0, 10.0, 1.0)
        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("Customer Profile")
        satisfaction_score = st.slider('Satisfaction Score', 1, 5, 3)
        city_tier = st.selectbox('City Tier', [1, 2, 3])
        gender = st.selectbox('Gender', ['Female', 'Male'])
        marital_status = st.selectbox('Marital Status', ['Single', 'Divorced', 'Married'])
        tenure_group = st.selectbox('Tenure Group', ['Early', 'New', 'MidTerm', 'LongTerm'])
        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("Behavioral & Preferences")
        preferred_login_device = st.selectbox('Login Device', ['Mobile Phone', 'Computer'])
        preferred_payment_mode = st.selectbox('Payment Mode', ['Debit Card', 'UPI', 'Credit Card', 'Cash on Delivery', 'E wallet'])
        preferred_order_cat = st.selectbox('Preferred Category', ['Laptop & Accessory', 'Mobile Phone', 'Others', 'Fashion', 'Grocery'])

        warehouse_to_home = st.number_input('Distance to Warehouse (km)', 0.0, 100.0, 10.0)
        number_of_address = st.number_input('Number of Addresses', 1, 10, 1)
        order_amount_hike = st.number_input('Order Amount Hike (%)', 0.0, 100.0, 10.0)
        coupon_used = st.number_input('Coupons Used', 0, 20, 0)
        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("Feedback & Status")
        complain = st.selectbox('Has Complaints?', ['No', 'Yes']) == 'Yes'
        unhappy_customer = st.selectbox('Unhappy Customer?', ['No', 'Yes']) == 'Yes'
        is_active_user = st.selectbox('Active User?', ['No', 'Yes']) == 'Yes'
        st.markdown("<br>", unsafe_allow_html=True)
    
    # --- FORMAT INPUT ---
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
        'Complain': int(complain),
        'OrderAmountHikeFromlastYear': order_amount_hike,
        'CouponUsed': coupon_used,
        'OrderCount': order_count,
        'DaySinceLastOrder': days_since_last_order,
        'CashbackAmount': cashback_amount,
        'RecencyRatio': recency_ratio,
        'IsActiveUser': int(is_active_user),
        'UnhappyCustomer': int(unhappy_customer),
        'TenureGroup': tenure_group
    }

    input_df = pd.DataFrame([input_dict])

    # --- PREDICTION ---
    if st.button('Predict'):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        proba_pct = proba * 100
        stay_pct = (1 - proba) * 100

        if prediction == 1:
            st.error(f"Churn Prediction: **Yes** (Probability: {proba_pct:.2f}%)")
            st.markdown("**Action Needed:** This customer is at risk of leaving. Consider targeted engagement.")
        else:
            st.success(f"Churn Prediction: **No** (Probability of Staying: {stay_pct:.2f}%)")
            st.markdown("**Good News:** This customer is likely to remain loyal.")

# --- IMAGE ---
st.image("ecommercepict.png", use_container_width=True)

# --- ABOUT ---
with st.expander("ℹ️ About This App"):
    st.markdown("""
This prediction tool helps e-commerce teams **identify high-risk customers** based on their behavior, satisfaction, and engagement level.

- Built using a supervised ML model trained on real e-commerce customer data.
- Use this tool to **prioritize retention campaigns**, improve support, and target offers.

**Churn** here means the customer is unlikely to continue using the app or shopping on the platform.
    """)
