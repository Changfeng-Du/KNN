import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# 加载数据
@st.cache_data
def load_data():
    dev = pd.read_csv('dev.csv')
    vad = pd.read_csv('vad.csv')
    return dev, vad

dev, vad = load_data()

# 读取 KNN 预测概率的 CSV 文件
@st.cache_data
def load_knn_probabilities():
    return pd.read_csv('test_probe.csv')

knn_probs = load_knn_probabilities()

# Streamlit 界面
st.title("Co-occurrence Risk Predictor (KNN Model)")

# 创建输入表单
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        smoker = st.selectbox("Smoker:", [1, 2, 3], 
                            format_func=lambda x: "Never" if x == 1 else "Former" if x == 2 else "Current")
        sex = st.selectbox("Sex:", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
        carace = st.selectbox("Race:", [1, 2, 3, 4, 5], 
                            format_func=lambda x: ["Mexican", "Hispanic", "White", "Black", "Other"][x - 1])
        drink = st.selectbox("Alcohol:", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
        sleep = st.selectbox("Sleep:", [1, 2], format_func=lambda x: "Problem" if x == 1 else "Normal")
        
    with col2:
        Hypertension = st.selectbox("Hypertension:", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
        Dyslipidemia = st.selectbox("Dyslipidemia:", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
        HHR = st.number_input("HHR Ratio:", min_value=0.2, max_value=1.7, value=1.0)
        RIDAGEYR = st.number_input("Age:", min_value=20, max_value=80, value=50)
        INDFMPIR = st.number_input("Poverty Ratio:", min_value=0.0, max_value=5.0, value=2.0)
        BMXBMI = st.number_input("BMI:", min_value=15.0, max_value=60.0, value=25.0)
        LBXWBCSI = st.number_input("WBC:", min_value=2.0, max_value=20.0, value=6.0)
        LBXRBCSI = st.number_input("RBC:", min_value=2.5, max_value=7.0, value=4.0)

    submitted = st.form_submit_button("Predict")

# 特征顺序（确保与模型训练时的特征顺序一致）
feature_names = [
    'smoker', 'sex', 'carace', 'drink', 'sleep',
    'Hypertension', 'Dyslipidemia', 'HHR', 'RIDAGEYR',
    'INDFMPIR', 'BMXBMI', 'LBXWBCSI', 'LBXRBCSI'
]

# 预测函数
def predict_class(input_df, knn_probs):
    # 在 CSV 文件中找到与输入数据匹配的预测概率
    input_tuple = tuple(input_df.iloc[0])
    matching_row = knn_probs[knn_probs[feature_names].apply(tuple, axis=1) == input_tuple]
    
    if matching_row.empty:
        return None  # 没有匹配的行，返回 None
    
    # 假设概率列是 'Yes' 和 'No'，可以根据具体的 CSV 结构修改
    prob_1 = matching_row['Yes'].values[0]
    prob_0 = 1 - prob_1
    predicted_class = 1 if prob_1 > 0.56 else 0
    return prob_1, prob_0, predicted_class

if submitted:
    # 构建输入数据
    input_data = [
        smoker, sex, carace, drink, sleep,
        Hypertension, Dyslipidemia, HHR, RIDAGEYR,
        INDFMPIR, BMXBMI, LBXWBCSI, LBXRBCSI
    ]
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    try:
        # 执行预测
        prob_1, prob_0, predicted_class = predict_class(input_df, knn_probs)
        
        if prob_1 is None:
            st.error("No matching record found in the prediction probabilities.")
            st.stop()
        
        # 显示结果
        st.success("### Prediction Results")
        st.metric("Comorbidity Risk", f"{prob_1 * 100:.1f}%", 
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
        background = shap.sample(vad[feature_names], 100)
        
        # 定义 SHAP 预测函数
        def shap_predict(data):
            input_df = pd.DataFrame(data, columns=feature_names)
            return np.column_stack([1 - prob_1, prob_1])  # 返回两个类别的概率
        
        # 创建解释器
        explainer = shap.KernelExplainer(shap_predict, background)
        shap_values = explainer.shap_values(input_df, nsamples=100)
        
        # 可视化
        st.subheader("Feature Impact")
        fig, ax = plt.subplots()
        shap.force_plot(explainer.expected_value[1], 
                       shap_values[0][:, 1], 
                       input_df.iloc[0],
                       matplotlib=True,
                       show=False)
        st.pyplot(fig)

        # LIME解释
        lime_exp = LimeTabularExplainer(
            background.values,
            feature_names=feature_names,
            class_names=['Low Risk', 'High Risk'],
            mode='classification'
        ).explain_instance(input_df.values[0], 
                           lambda x: np.column_stack([1 - prob_1, prob_1]))
        
        st.components.v1.html(lime_exp.as_html(), height=800)

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.stop()
