import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Load the new model
@st.cache_resource
def load_model():
    model = joblib.load('knn_model.pkl')  # Load the .pkl model instead of .rds
    return model

model = load_model()

# Load data
@st.cache_data
def load_data():
    dev = pd.read_csv('dev.csv')
    vad = pd.read_csv('vad.csv')
    return dev, vad

dev, vad = load_data()

# Define feature names from the dataset
feature_names = [
    'smoker', 'sex', 'carace', 'drink', 'sleep',
    'Hypertension', 'Dyslipidemia', 'HHR', 'RIDAGEYR',
    'INDFMPIR', 'BMXBMI', 'LBXWBCSI', 'LBXRBCSI'
]

# Streamlit interface
st.title("Co-occurrence Risk Predictor (Pkl Model)")

# Create form for input fields
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        smoker = st.selectbox("Smoker:", [1, 2, 3], 
                            format_func=lambda x: "Never" if x==1 else "Former" if x==2 else "Current")
        sex = st.selectbox("Sex:", [1, 2], format_func=lambda x: "Female" if x==1 else "Male")
        carace = st.selectbox("Race:", [1, 2, 3, 4, 5], 
                            format_func=lambda x: ["Mexican","Hispanic","White","Black","Other"][x-1])
        drink = st.selectbox("Alcohol:", [1, 2], format_func=lambda x: "No" if x==1 else "Yes")
        sleep = st.selectbox("Sleep:", [1, 2], format_func=lambda x: "Problem" if x==1 else "Normal")
        
    with col2:
        Hypertension = st.selectbox("Hypertension:", [1, 2], format_func=lambda x: "No" if x==1 else "Yes")
        Dyslipidemia = st.selectbox("Dyslipidemia:", [1, 2], format_func=lambda x: "No" if x==1 else "Yes")
        HHR = st.number_input("HHR Ratio:", min_value=0.2, max_value=1.7, value=1.0)
        RIDAGEYR = st.number_input("Age:", min_value=20, max_value=80, value=50)
        INDFMPIR = st.number_input("Poverty Ratio:", min_value=0.0, max_value=5.0, value=2.0)
        BMXBMI = st.number_input("BMI:", min_value=15.0, max_value=60.0, value=25.0)
        LBXWBCSI = st.number_input("WBC:", min_value=2.0, max_value=20.0, value=6.0)
        LBXRBCSI = st.number_input("RBC:", min_value=2.5, max_value=7.0, value=4.0)

    submitted = st.form_submit_button("Predict")

# Prediction function using the loaded .pkl model
def predict(input_df):
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)[:,1]  # Probabilities for class 1
    return prediction, prob

if submitted:
    # Construct the input dataframe
    input_data = [
        smoker, sex, carace, drink, sleep,
        Hypertension, Dyslipidemia, HHR, RIDAGEYR,
        INDFMPIR, BMXBMI, LBXWBCSI, LBXRBCSI
    ]
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    try:
        # Get the prediction and probabilities
        pred_class, prob = predict(input_df)
        prob_1 = prob[0]
        prob_0 = 1 - prob_1
        predicted_class = 1 if prob_1 > 0.56 else 0
        
        # Display the prediction result
        st.success("### Prediction Results")
        st.metric("Comorbidity Risk", f"{prob_1*100:.1f}%", 
                  help="Probability of having both conditions")
        
        # Generate advice based on the prediction
        advice_template = """
        **Recommendations:**
        - Regular cardiovascular screening
        - Monitor blood pressure weekly
        - Mediterranean diet recommended
        {}"""
        st.info(advice_template.format("Immediate consultation needed!" if predicted_class == 1 else "Maintain healthy lifestyle"))

        # SHAP explanation
        st.subheader("Model Interpretation")

        # Prepare background data for SHAP
        background = shap.sample(vad[feature_names], 100)
        
        # Define SHAP prediction function
        def shap_predict(data):
            input_df = pd.DataFrame(data, columns=feature_names)
            return np.column_stack([1-predict(input_df)[1], predict(input_df)[1]])
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(shap_predict, background)
        shap_values = explainer.shap_values(input_df, nsamples=100)
        
        # Visualize SHAP values
        st.subheader("Feature Impact")
        fig, ax = plt.subplots()
        shap.force_plot(explainer.expected_value[1], 
                       shap_values[0][:,1], 
                       input_df.iloc[0],
                       matplotlib=True,
                       show=False)
        st.pyplot(fig)

        # LIME explanation
        lime_exp = LimeTabularExplainer(
            background.values,
            feature_names=feature_names,
            class_names=['Low Risk', 'High Risk'],
            mode='classification'
        ).explain_instance(input_df.values[0], 
                           lambda x: np.column_stack([1-predict(pd.DataFrame(x, columns=feature_names))[1],
                                                     predict(pd.DataFrame(x, columns=feature_names))[1]]))
        
        st.components.v1.html(lime_exp.as_html(), height=800)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()
