import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from lime.lime_tabular import LimeTabularExplainer

# 加载数据
@st.cache_data
def load_data():
    dev = pd.read_csv('dev.csv')
    vad = pd.read_csv('vad.csv')
    prob_df = pd.read_csv('test_probe.csv')  # 加载预生成的预测概率
    return dev, vad, prob_df

dev, vad, prob_df = load_data()

# 定义特征顺序（必须与模型训练时完全一致）
feature_names = [
    'smoker', 'sex', 'carace', 'drink', 'sleep',
    'Hypertension', 'Dyslipidemia', 'HHR', 'RIDAGEYR',
    'INDFMPIR', 'BMXBMI', 'LBXWBCSI', 'LBXRBCSI'
]

# 初始化最近邻模型
@st.cache_resource
def init_nn_model():
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(prob_df[feature_names])
    return nn

nn_model = init_nn_model()

# Streamlit界面
st.title("Co-occurrence Risk Predictor (Precomputed)")

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

if submitted:
    # 构建输入数据
    input_data = [
        smoker, sex, carace, drink, sleep,
        Hypertension, Dyslipidemia, HHR, RIDAGEYR,
        INDFMPIR, BMXBMI, LBXWBCSI, LBXRBCSI
    ]
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    try:
        # 查找最近邻
        _, indices = nn_model.kneighbors(input_df)
        nearest_idx = indices[0][0]
        prob_1 = prob_df.iloc[nearest_idx]['pred_prob']
        
        # 显示结果
        st.success("### Prediction Results")
        st.metric("Comorbidity Risk", f"{prob_1*100:.1f}%", 
                help="Probability from nearest matching case in precomputed data")
        
        # 生成建议
        advice_template = """
        **Recommendations:**
        - Regular cardiovascular screening
        - Monitor blood pressure weekly
        - Mediterranean diet recommended
        {}"""
        st.info(advice_template.format("Immediate consultation needed!" if prob_1 > 0.56 else "Maintain healthy lifestyle"))

        # 解释部分（使用预存数据）
        st.subheader("Feature Comparison")
        
        # 显示匹配样本的特征对比
        matched_sample = prob_df.iloc[nearest_idx][feature_names]
        comparison_df = pd.DataFrame({
            'Your Input': input_df.iloc[0],
            'Matched Sample': matched_sample
        })
        st.dataframe(comparison_df.T.style.highlight_max(axis=1))

        # 简化版SHAP解释（使用预存数据）
        st.subheader("Feature Importance")
        fig, ax = plt.subplots()
        shap_values = prob_df.iloc[nearest_idx][feature_names].values - prob_df[feature_names].mean().values
        shap.summary_plot(shap_values.reshape(1, -1), 
                         feature_names=feature_names,
                         plot_type="bar",
                         show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.stop()
