import streamlit as st
import pickle as pkl
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
scal = MinMaxScaler()
model = pkl.load(open("rofh_model.p", "rb"))

# Function to preprocess user input
def preprocess(age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal):
    sex = 1 if sex == "male" else 0
    
    # Map chest pain type (cp) to numerical values
    cp_map = {
        "Typical angina": 0,
        "Atypical angina": 1,
        "Non-anginal pain": 2,
        "Asymptomatic": 2
    }
    cp = cp_map[cp]
    
    exang = 1 if exang == "Yes" else 0
    fbs = 1 if fbs == "Yes" else 0
    
    # Map heart rate slope (slope) to numerical values
    slope_map = {
        "Upsloping: better heart rate with exercise (uncommon)": 0,
        "Flatsloping: minimal change (typical healthy heart)": 1,
        "Downsloping: signs of unhealthy heart": 2
    }
    slope = slope_map[slope]
    
    # Map thalium stress test result (thal) to numerical values
    thal_map = {
        "fixed defect: used to be defect but ok now": 6,
        "reversable defect: no proper blood movement when exercising": 7,
        "normal": 2.31  # Assuming this is a numeric value based on your previous code
    }
    thal = thal_map[thal]
    
    # Map resting electrocardiographic results (restecg) to numerical values
    restecg_map = {
        "Nothing to note": 0,
        "ST-T Wave abnormality": 1,
        "Possible or definite enlargement of the left ventricle": 2
    }
    restecg = restecg_map[restecg]

    user_input = [age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal]
    user_input = np.array(user_input).reshape(1, -1)
    user_input = scal.fit_transform(user_input)
    prediction = model.predict(user_input)

    return prediction

# Streamlit app configuration
st.set_page_config(page_title="Heart Disease Forecaster", page_icon="❤️‍🩹", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a beautiful design
st.markdown(
    """
    <style>
    body {
        background-color: #f9f9f9;
        color: #333333;
        font-family: Arial, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
        padding: 20px;
        border-radius: 10px;
    }
    .main {
        background-color: #ffffff;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
        padding: 20px;
        border-radius: 10px;
    }
    .header {
        background-color: #4285F4;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .header h1 {
        color: white;
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        color: #666666;
    }
    .footer a {
        color: #4285F4;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Frontend elements of the web page
st.markdown(
    """
    <div class="header">
    <h1>Heart Disease Forecaster 🫀</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Input fields for user data
with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("About App")
        st.markdown("This app helps you check your risk of developing heart disease.")
        st.markdown("Enter your information and click 'Predict' to see your result.")
        st.markdown("Don't forget to rate the app!")

    with col2:
        st.subheader("Input Your Data")
        age = st.selectbox("Age", range(1, 121, 1))
        sex = st.radio("Select Gender", ('male', 'female'))
        cp = st.selectbox('Chest Pain Type', ("Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"))
        trestbps = st.selectbox('Resting Blood Sugar', range(1, 500, 1))
        restecg = st.selectbox('Resting Electrocardiographic Results',
                               ("Nothing to note", "ST-T Wave abnormality", "Possible or definite left ventricular hypertrophy"))
        chol = st.selectbox('Serum Cholestoral in mg/dl', range(1, 1000, 1))
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ['Yes', 'No'])
        thalach = st.selectbox('Maximum Heart Rate Achieved', range(1, 300, 1))
        exang = st.selectbox('Exercise Induced Angina', ["Yes", "No"])
        oldpeak = st.number_input('Oldpeak')
        slope = st.selectbox('Heart Rate Slope',
                             ("Upsloping: better heart rate with exercise (uncommon)",
                              "Flatsloping: minimal change (typical healthy heart)",
                              "Downsloping: signs of unhealthy heart"))
        ca = st.selectbox('Number of Major Vessels Colored by Flourosopy', range(0, 5, 1))
        thal = st.selectbox('Thalium Stress Result', ("fixed defect: used to be defect but ok now",
                                                      "reversable defect: no proper blood movement when exercising",
                                                      "normal"))

# Processing user input and making prediction
pred = preprocess(age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal)

# Predict button
if st.button("Forecast", key="predict_button"):
    with st.spinner('Forecasting...'):
        if pred[0] == 0:
            st.error('**Warning!** You have a high risk of getting a heart attack!')
        else:
            st.success('**You have a lower risk** of getting a heart disease!')

# Footer
st.markdown(
    """
    <div class="footer">
    <p>Supported by Rofhiwa and Eduvos Data Science Hons</p>
    <p>Note: This app provides forecasting, not medical advice. See a doctor if you have persistent symptoms.</p>
    </div>
    """,
    unsafe_allow_html=True
)
