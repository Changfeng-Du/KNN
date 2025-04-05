import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# 初始化R环境
pandas2ri.activate()
robjects.r['options'](warn=-1)

# 加载R模型
@st.cache_resource
def load_r_model():
    r_model = robjects.r['readRDS']('knn_model.rds')
    
    # 动态修补Pandas兼容性
    if not hasattr(pd.DataFrame, 'iteritems'):
        pd.DataFrame.iteritems = pd.DataFrame.items
        
    return r_model

r_model = load_r_model()

# 加载数据
@st.cache_data
def load_data():
    dev = pd.read_csv('dev.csv')
    vad = pd.read_csv('vad.csv')
    return dev, vad

dev, vad = load_data()

# 加载训练集和验证集的预测概率数据
@st.cache_data
def load_pred_probabilities():
    train_probe = pd.read_csv('train_probe.csv')
    test_probe = pd.read_csv('test_probe.csv')
    return train_probe, test_probe

train_probe, test_probe = load_pred_probabilities()

# 定义特征顺序（根据实际数据调整）
feature_names = [
    'smoker', 'sex', 'carace', 'drink', 'sleep',
    'Hypertension', 'Dyslipidemia', 'HHR', 'RIDAGEYR',
    'INDFMPIR', 'BMXBMI', 'LBXWBCSI', 'LBXRBCSI'
]

# Streamlit界面
st.title("Co-occurrence Risk Predictor (R Model)")

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

# 预测函数 (基于概率文件)
def predict_from_probabilities(input_data, probability_data):
    # 由于输入数据和概率数据没有直接对应的关系，需要根据输入值与预测数据找到对应的概率
    # 假设你是通过一个最近邻匹配方法来找到最接近的记录
    pred_data = probability_data[['No', 'Yes']]
    # 使用输入数据的索引来查找对应的预测概率
    idx = probability_data[probability_data[feature_names].eq(input_data).all(axis=1)].index[0]
    
    prob_1 = pred_data.loc[idx, 'Yes']
    prob_0 = 1 - prob_1
    predicted_class = 1 if prob_1 > 0.56 else 0
    
    return prob_1, predicted_class

if submitted:
    # 构建输入数据
    input_data = [
        smoker, sex, carace, drink, sleep,
        Hypertension, Dyslipidemia, HHR, RIDAGEYR,
        INDFMPIR, BMXBMI, LBXWBCSI, LBXRBCSI
    ]
    
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # 进行预测
    try:
        prob_1, predicted_class = predict_from_probabilities(input_data, test_probe)

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

        # SHAP解释
        st.subheader("Model Interpretation")
        
        # 准备解释数据
        background = shap.sample(test_probe[feature_names], 100)
        
        # 定义SHAP预测函数
        def shap_predict(data):
            input_df = pd.DataFrame(data, columns=feature_names)
            return np.column_stack([1 - predict_from_probabilities(input_df, test_probe)[0], 
                                    predict_from_probabilities(input_df, test_probe)[0]])
        
        # 创建解释器
        explainer = shap.KernelExplainer(shap_predict, background)
        shap_values = explainer.shap_values(input_data, nsamples=100)
        
        # 可视化
        st.subheader("Feature Impact")
        fig, ax = plt.subplots()
        shap.force_plot(explainer.expected_value[1], 
                        shap_values[0][:,1], 
                        input_data,
                        matplotlib=True,
                        show=False)
        st.pyplot(fig)

        # LIME解释
        lime_exp = LimeTabularExplainer(
            background.values,
            feature_names=feature_names,
            class_names=['Low Risk', 'High Risk'],
            mode='classification'
        ).explain_instance(input_data, 
                           lambda x: np.column_stack([1 - predict_from_probabilities(pd.DataFrame(x, columns=feature_names), test_probe)[0], 
                                                      predict_from_probabilities(pd.DataFrame(x, columns=feature_names), test_probe)[0]]))
        
        st.components.v1.html(lime_exp.as_html(), height=800)

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.stop()
