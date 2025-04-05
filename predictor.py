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

# 定义特征顺序（根据实际数据调整）
feature_names = [
    'smoker', 'sex', 'carace', 'drink', 'sleep',
    'Hypertension', 'Dyslipidemia', 'HHR', 'RIDAGEYR',
    'INDFMPIR', 'BMXBMI', 'LBXWBCSI', 'LBXRBCSI'
]

# 修改后的预测函数
def r_predict(input_df):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_data = robjects.conversion.py2rpy(input_df)
    
    predict_r = robjects.r['predict']
    
    try:
        # 尝试获取概率预测
        r_pred_prob = predict_r(r_model, newdata=r_data, type="prob")
        
        # 将R对象转换为Python对象
        with localconverter(robjects.default_converter + pandas2ri.converter):
            pred_prob = robjects.conversion.rpy2py(r_pred_prob)
        
        # 处理不同概率返回格式
        if isinstance(pred_prob, pd.DataFrame):
            # DataFrame格式（包含'Yes'列）
            if 'Yes' in pred_prob.columns:
                prob_yes = pred_prob['Yes'].values
            else:
                # 假设第二列是阳性概率
                prob_yes = pred_prob.iloc[:, 1].values
        elif isinstance(pred_prob, np.ndarray):
            # Numpy数组格式（假设第二列是阳性概率）
            prob_yes = pred_prob[:, 1] if pred_prob.ndim == 2 else pred_prob
        else:
            raise ValueError("Unsupported probability format")
            
        return prob_yes
    
    except Exception as e:
        st.warning(f"Probability prediction failed: {str(e)}, attempting class prediction...")
        try:
            # 获取原始类别预测
            r_pred_class = predict_r(r_model, newdata=r_data)
            y_pred = np.array(r_pred_class).flatten().astype(int)
            
            # 处理R的1/2编码
            if set(y_pred) == {1, 2}:
                y_pred = y_pred - 1
            
            # 返回模拟概率（1表示100%，0表示0%）
            return y_pred.astype(float)
        
        except Exception as class_error:
            st.error(f"Prediction failed completely: {str(class_error)}")
            raise

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
        prob_1 = r_predict(input_df)[0]
        prob_0 = 1 - prob_1
        # 使用代码1中确定的阈值
        predicted_class = 1 if prob_1 > 0.626107357806162 else 0
        
        # 显示结果
        st.success("### Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Comorbidity Risk", 
                      f"{prob_1*100:.1f}%", 
                      delta=f"{'High' if predicted_class==1 else 'Low'} risk",
                      help="Probability of having both conditions")
        with col2:
            st.metric("Decision Threshold", 
                      "62.6%", 
                      help="Cut-off value for risk classification")
        
        # 生成建议
        advice_template = """
        **Recommendations:**
        - Regular cardiovascular screening
        - Monitor blood pressure weekly
        - Mediterranean diet recommended
        {}"""
        additional_advice = "Immediate consultation recommended!" if predicted_class==1 else "Maintain current healthy lifestyle"
        st.info(advice_template.format(additional_advice))

        # SHAP解释
        st.subheader("Model Interpretation")
        
        # 准备解释数据
        background = shap.sample(vad[feature_names], 100)
        
        # 定义SHAP预测函数
        def shap_predict(data):
            input_df = pd.DataFrame(data, columns=feature_names)
            prob = r_predict(input_df)
            return np.column_stack([1-prob, prob])
        
        # 创建解释器
        explainer = shap.KernelExplainer(shap_predict, background)
        shap_values = explainer.shap_values(input_df, nsamples=100)
        
        # 可视化
        st.subheader("Feature Impact")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.force_plot(explainer.expected_value[1], 
                       shap_values[1][0],  # 使用阳性类的SHAP值
                       input_df.iloc[0],
                       matplotlib=True,
                       show=False)
        st.pyplot(fig)
        
        # LIME解释
        lime_exp = LimeTabularExplainer(
            background.values,
            feature_names=feature_names,
            class_names=['Low Risk','High Risk'],
            mode='classification'
        ).explain_instance(
            input_df.values[0], 
            lambda x: np.column_stack((1-r_predict(pd.DataFrame(x, columns=feature_names)),
                                  r_predict(pd.DataFrame(x, columns=feature_names)))),
            num_features=len(feature_names)
        
        st.components.v1.html(lime_exp.as_html(), height=800)

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.stop()
