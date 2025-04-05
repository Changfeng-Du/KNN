import streamlit as st
import pandas as pd
import numpy as np

# 加载KNN预测概率的CSV文件
@st.cache_data
def load_knn_probabilities():
    knn_probs = pd.read_csv('test_probe.csv')
    return knn_probs

# 加载数据
@st.cache_data
def load_data():
    dev = pd.read_csv('dev.csv')
    vad = pd.read_csv('vad.csv')
    return dev, vad

# Streamlit界面
st.title("Co-occurrence Risk Predictor (KNN Model)")

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

# 预测函数
def get_knn_prediction(input_data, knn_probs):
    # 这里根据输入的数据去寻找相似的预测概率
    # 假设knn_probs中每行是一个预测概率，你需要根据实际情况选择最接近的行
    # 举个例子，假设我们选择 target 最接近的预测行
    pred_prob = knn_probs.loc[knn_probs['target'] == input_data[0], ['No', 'Yes']].values
    if pred_prob.size == 0:
        return None  # 如果没有找到合适的预测行
    return pred_prob[0]

if submitted:
    # 构建输入数据（这里仅使用 target 列作为输入数据）
    input_data = [smoker, sex, carace, drink, sleep, Hypertension, Dyslipidemia, HHR, RIDAGEYR, INDFMPIR, BMXBMI, LBXWBCSI, LBXRBCSI]
    
    # 加载KNN预测概率
    knn_probs = load_knn_probabilities()

    # 执行预测
    pred_prob = get_knn_prediction(input_data, knn_probs)
    
    if pred_prob is not None:
        prob_1 = pred_prob[1]  # 'Yes'类的概率
        prob_0 = pred_prob[0]  # 'No'类的概率
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
        st.info(advice_template.format("Immediate consultation needed!" if predicted_class == 1 else "Maintain healthy lifestyle"))
    else:
        st.error("No matching prediction found in the KNN model data.")
