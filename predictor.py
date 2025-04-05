import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# 加载新的模型
@st.cache_resource
def load_model():
    model = joblib.load('knn_model.pkl')  # 加载 .pkl 模型而不是 .rds
    return model

model = load_model()

# 加载数据
@st.cache_data
def load_data():
    dev = pd.read_csv('dev.csv')
    vad = pd.read_csv('vad.csv')
    return dev, vad

dev, vad = load_data()

# 定义数据集的特征名称
feature_names = [
    'smoker', 'sex', 'carace', 'drink', 'sleep',
    'Hypertension', 'Dyslipidemia', 'HHR', 'RIDAGEYR',
    'INDFMPIR', 'BMXBMI', 'LBXWBCSI', 'LBXRBCSI'
]

# Streamlit 界面
st.title("共病风险预测器 (Pkl 模型)")

# 创建输入字段表单
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        smoker = st.selectbox("吸烟者:", [1, 2, 3], 
                            format_func=lambda x: "从不" if x==1 else "曾经" if x==2 else "当前")
        sex = st.selectbox("性别:", [1, 2], format_func=lambda x: "女性" if x==1 else "男性")
        carace = st.selectbox("种族:", [1, 2, 3, 4, 5], 
                            format_func=lambda x: ["墨西哥人", "西班牙裔", "白人", "黑人", "其他"][x-1])
        drink = st.selectbox("饮酒:", [1, 2], format_func=lambda x: "否" if x==1 else "是")
        sleep = st.selectbox("睡眠:", [1, 2], format_func=lambda x: "问题" if x==1 else "正常")
        
    with col2:
        Hypertension = st.selectbox("高血压:", [1, 2], format_func=lambda x: "否" if x==1 else "是")
        Dyslipidemia = st.selectbox("血脂异常:", [1, 2], format_func=lambda x: "否" if x==1 else "是")
        HHR = st.number_input("HHR 比率:", min_value=0.2, max_value=1.7, value=1.0)
        RIDAGEYR = st.number_input("年龄:", min_value=20, max_value=80, value=50)
        INDFMPIR = st.number_input("贫困比率:", min_value=0.0, max_value=5.0, value=2.0)
        BMXBMI = st.number_input("BMI:", min_value=15.0, max_value=60.0, value=25.0)
        LBXWBCSI = st.number_input("WBC:", min_value=2.0, max_value=20.0, value=6.0)
        LBXRBCSI = st.number_input("RBC:", min_value=2.5, max_value=7.0, value=4.0)

    submitted = st.form_submit_button("预测")

# 使用加载的 .pkl 模型进行预测
def predict(input_df):
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)[:,1]  # 类别1的概率
    return prediction, prob

if submitted:
    # 构建输入数据框
    input_data = [
        smoker, sex, carace, drink, sleep,
        Hypertension, Dyslipidemia, HHR, RIDAGEYR,
        INDFMPIR, BMXBMI, LBXWBCSI, LBXRBCSI
    ]
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    try:
        # 获取预测结果和概率
        pred_class, prob = predict(input_df)
        prob_1 = prob[0]
        prob_0 = 1 - prob_1
        predicted_class = 1 if prob_1 > 0.56 else 0
        
        # 显示预测结果
        st.success("### 预测结果")
        st.metric("共病风险", f"{prob_1*100:.1f}%", 
                  help="患有两种疾病的概率")
        
        # 根据预测结果生成建议
        advice_template = """
        **建议:**
        - 定期进行心血管筛查
        - 每周监测血压
        - 推荐地中海饮食
        {}"""
        st.info(advice_template.format("建议立即咨询医生!" if predicted_class == 1 else "保持健康的生活方式"))

        # SHAP 解释
        st.subheader("模型解释")

        # 准备 SHAP 背景数据
        background = shap.sample(vad[feature_names], 100)
        
        # 定义 SHAP 预测函数
        def shap_predict(data):
            input_df = pd.DataFrame(data, columns=feature_names)
            return np.column_stack([1-predict(input_df)[1], predict(input_df)[1]])
        
        # 创建 SHAP 解释器
        explainer = shap.KernelExplainer(shap_predict, background)
        shap_values = explainer.shap_values(input_df, nsamples=100)
        
        # 可视化 SHAP 值
        st.subheader("特征影响")
        fig, ax = plt.subplots()
        shap.force_plot(explainer.expected_value[1], 
                       shap_values[0][:,1], 
                       input_df.iloc[0],
                       matplotlib=True,
                       show=False)
        st.pyplot(fig)

        # LIME 解释
        lime_exp = LimeTabularExplainer(
            background.values,
            feature_names=feature_names,
            class_names=['低风险', '高风险'],
            mode='classification'
        ).explain_instance(input_df.values[0], 
                           lambda x: np.column_stack([1-predict(pd.DataFrame(x, columns=feature_names))[1],
                                                     predict(pd.DataFrame(x, columns=feature_names))[1]]))
        
        st.components.v1.html(lime_exp.as_html(), height=800)

    except Exception as e:
        st.error(f"预测错误: {str(e)}")
        st.stop()
