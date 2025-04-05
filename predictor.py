import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# 加载训练集和验证集的预测概率数据
@st.cache_data
def load_probability_data():
    train_probs = pd.read_csv('train_probe.csv')  # 假设是训练集的预测概率
    val_probs = pd.read_csv('test_probe.csv')      # 假设是验证集的预测概率
    return train_probs, val_probs

train_probs, val_probs = load_probability_data()

# 定义特征顺序（根据实际数据调整）
feature_names = [
    'smoker', 'sex', 'carace', 'drink', 'sleep',
    'Hypertension', 'Dyslipidemia', 'HHR', 'RIDAGEYR',
    'INDFMPIR', 'BMXBMI', 'LBXWBCSI', 'LBXRBCSI'
]

# Streamlit界面
st.title("Co-occurrence Risk Predictor")

# 创建输入表单
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        smoker = st.selectbox("Smoker:", [1,2,3], 
                            format_func=lambda x: "Never" if x==1 else "Former" if x==2 else "Current")
        sex = st.selectbox("Sex:", [1,2], format_func=lambda x: "Female" if x==1 else "Male")
        carace = st.selectbox("Race:", [1,2,3,4,5], 
                            format_func=lambda x: ["Mexican","Hispanic","White","Black","Other"][x-1])
        drink = st.selectbox("Alcohol:", [1,2], format_func=lambda x: "No" if x==1 else "Yes")
        sleep = st.selectbox("Sleep:", [1,2], format_func=lambda x: "Problem" if x==1 else "Normal")
        
    with col2:
        Hypertension = st.selectbox("Hypertension:", [1,2], format_func=lambda x: "No" if x==1 else "Yes")
        Dyslipidemia = st.selectbox("Dyslipidemia:", [1,2], format_func=lambda x: "No" if x==1 else "Yes")
        HHR = st.number_input("HHR Ratio:", min_value=0.2, max_value=1.7, value=1.0)
        RIDAGEYR = st.number_input("Age:", min_value=20, max_value=80, value=50)
        INDFMPIR = st.number_input("Poverty Ratio:", min_value=0.0, max_value=5.0, value=2.0)
        BMXBMI = st.number_input("BMI:", min_value=15.0, max_value=60.0, value=25.0)
        LBXWBCSI = st.number_input("WBC:", min_value=2.0, max_value=20.0, value=6.0)
        LBXRBCSI = st.number_input("RBC:", min_value=2.5, max_value=7.0, value=4.0)

    submitted = st.form_submit_button("Predict")

# 预测函数，基于训练集的预测概率数据来做预测
def predict_from_probabilities(input_data, train_probs):
    # 假设 `train_probs` 包含了训练集样本和对应的预测概率
    # 我们可以根据输入的特征计算最接近的样本，并返回其预测概率
    # 这里简单实现一个基于输入数据与训练集特征的相似度计算方法
    # 对于示范，我们直接用`train_probs`中的概率数据来预测
    # 您可以根据具体情况调整
    closest_row_idx = np.argmin(np.sum(np.abs(train_probs[feature_names] - input_data), axis=1))
    return train_probs.iloc[closest_row_idx]['Yes']

if submitted:
    # 构建输入数据
    input_data = [
        smoker, sex, carace, drink, sleep,
        Hypertension, Dyslipidemia, HHR, RIDAGEYR,
        INDFMPIR, BMXBMI, LBXWBCSI, LBXRBCSI
    ]
    
    # 转化为DataFrame来与训练集数据比较
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    try:
        # 使用训练集概率数据进行预测
        prob_1 = predict_from_probabilities(input_data, train_probs)
        prob_0 = 1 - prob_1
        predicted_class = 1 if prob_1 > 0.56 else 0
        
        # 显示结果
        st.success("### Prediction Results")
        st.metric("Comorbidity Risk", f"{prob_1*100:.1f}%", 
                help="Probability of having both conditions")
        
        # 生成建议
        advice_template = """
        **Recommendations:**
        - Regular cardiovascular screening
        - Monitor blood pressure weekly
        - Mediterranean diet recommended
        {}"""
        st.info(advice_template.format("Immediate consultation needed!" if predicted_class==1 else "Maintain healthy lifestyle"))

        # SHAP解释
        st.subheader("Model Interpretation")
        
        # 准备解释数据
        background = shap.sample(val_probs[feature_names], 100)
        
        # 定义SHAP预测函数
        def shap_predict(data):
            input_df = pd.DataFrame(data, columns=feature_names)
            return np.column_stack([1-predict_from_probabilities(input_df), predict_from_probabilities(input_df)])
        
        # 创建解释器
        explainer = shap.KernelExplainer(shap_predict, background)
        shap_values = explainer.shap_values(input_df, nsamples=100)
        
        # 可视化
        st.subheader("Feature Impact")
        fig, ax = plt.subplots()
        shap.force_plot(explainer.expected_value[1], 
                       shap_values[0][:,1], 
                       input_df.iloc[0],
                       matplotlib=True,
                       show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.stop()
