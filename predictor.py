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
robjects.r('''
    library(caret)
    predict_caret <- function(model, newdata) {
        # 确保特征类型与训练时一致
        factor_cols <- c("smoker", "sex", "carace", "drink", "sleep", 
                       "Hypertension", "Dyslipidemia")
        newdata[factor_cols] <- lapply(newdata[factor_cols], factor)
        
        # 生成预测概率
        pred_probs <- predict(model, newdata = newdata, type = "prob")
        if ("Yes" %in% colnames(pred_probs)) {
            return(pred_probs$Yes)
        } else {
            return(pred_probs[,2])  # 兼容不同版本
        }
    }
''')

# 加载R模型
@st.cache_resource
def load_r_model():
    r_model = robjects.r['readRDS']('knn_model.rds')
    return r_model

r_model = load_r_model()

# 加载数据
@st.cache_data
def load_data():
    dev = pd.read_csv('dev.csv')
    vad = pd.read_csv('vad.csv')
    return dev, vad

dev, vad = load_data()

# 定义特征顺序和元数据
feature_metadata = {
    'smoker': {
        'options': [1, 2, 3],
        'labels': ["Never", "Former", "Current"]
    },
    'sex': {
        'options': [1, 2],
        'labels': ["Female", "Male"]
    },
    'carace': {
        'options': [1, 2, 3, 4, 5],
        'labels': ["Mexican", "Hispanic", "White", "Black", "Other"]
    },
    # 其他分类变量类似定义...
}

feature_names = list(feature_metadata.keys()) + [
    'HHR', 'RIDAGEYR', 'INDFMPIR', 'BMXBMI', 'LBXWBCSI', 'LBXRBCSI'
]

# Streamlit界面
st.title("Co-occurrence Risk Predictor (R Model)")

# 创建输入表单
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    input_values = {}
    with col1:
        for feature in list(feature_metadata.keys())[:5]:
            meta = feature_metadata[feature]
            input_values[feature] = st.selectbox(
                f"{feature.capitalize()}:",
                options=meta['options'],
                format_func=lambda x, m=meta: m['labels'][m['options'].index(x)]
            )
    
    with col2:
        for feature in list(feature_metadata.keys())[5:7]:
            meta = feature_metadata[feature]
            input_values[feature] = st.selectbox(
                f"{feature.capitalize()}:",
                options=meta['options'],
                format_func=lambda x, m=meta: m['labels'][m['options'].index(x)]
            )
        
        input_values['HHR'] = st.number_input("HHR Ratio:", min_value=0.2, max_value=1.7, value=1.0)
        input_values['RIDAGEYR'] = st.number_input("Age:", min_value=20, max_value=80, value=50)
        input_values['INDFMPIR'] = st.number_input("Poverty Ratio:", min_value=0.0, max_value=5.0, value=2.0)
        input_values['BMXBMI'] = st.number_input("BMI:", min_value=15.0, max_value=60.0, value=25.0)
        input_values['LBXWBCSI'] = st.number_input("WBC:", min_value=2.0, max_value=20.0, value=6.0)
        input_values['LBXRBCSI'] = st.number_input("RBC:", min_value=2.5, max_value=7.0, value=4.0)

    submitted = st.form_submit_button("Predict")

# 预测函数
def r_predict(input_df):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_data = robjects.conversion.py2rpy(input_df[feature_names])
    
    r_pred = robjects.globalenv['predict_caret'](r_model, r_data)
    return np.array(r_pred)

if submitted:
    try:
        # 构建输入数据
        input_df = pd.DataFrame([input_values])[feature_names]
        
        # 执行预测
        prob_1 = r_predict(input_df)[0]
        prob_0 = 1 - prob_1
        predicted_class = 1 if prob_1 > 0.56 else 0
        
        # 显示结果
        st.success("### Prediction Results")
        st.metric("Comorbidity Risk", f"{prob_1*100:.1f}%", 
                delta=f"{(prob_1-0.56)*100:.1f}% vs threshold", 
                help="Probability of having both conditions")
        
        # 生成建议
        advice_template = """
        **Recommendations:**
        - Regular cardiovascular screening
        - Monitor blood pressure weekly
        - Mediterranean diet recommended
        {}"""
        st.info(advice_template.format("🚨 Immediate consultation needed!" if predicted_class==1 else "✅ Maintain healthy lifestyle"))

        # SHAP解释
        st.subheader("Model Interpretation")
        
        # 准备解释数据
        background = shap.sample(vad[feature_names], 100)
        
        # 定义SHAP预测函数
        def shap_predict(data):
            return np.column_stack([1 - r_predict(pd.DataFrame(data, columns=feature_names)),
                                   r_predict(pd.DataFrame(data, columns=feature_names))])
        
        # 创建解释器
        explainer = shap.KernelExplainer(shap_predict, background)
        shap_values = explainer.shap_values(input_df, nsamples=100)
        
        # 可视化
        fig, ax = plt.subplots()
        shap.force_plot(explainer.expected_value[1], 
                       shap_values[1][0,:], 
                       input_df.iloc[0],
                       feature_names=feature_names,
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
            shap_predict,
            num_features=10
        )
        st.components.v1.html(lime_exp.as_html(), height=800)

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.stop()

# 侧边栏显示数据摘要
st.sidebar.header("Data Summary")
st.sidebar.write("Training data shape:", dev.shape)
st.sidebar.write("Validation data shape:", vad.shape)
st.sidebar.write("Feature names:", feature_names)
