import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.metrics import roc_curve, auc

# 初始化R环境
pandas2ri.activate()
robjects.r['options'](warn=-1)

# 加载R模型和评估数据
@st.cache_resource
def load_resources():
    # 加载R模型
    r_model = robjects.r['readRDS']('knn_model.rds')
    
    # 加载评估数据
    train_probe = pd.read_csv('train_probe.csv', sep='\t')
    test_probe = pd.read_csv('test_probe.csv', sep='\t')
    
    # 动态修补Pandas兼容性
    if not hasattr(pd.DataFrame, 'iteritems'):
        pd.DataFrame.iteritems = pd.DataFrame.items
        
    return r_model, train_probe, test_probe

r_model, train_probe, test_probe = load_resources()

# 加载特征数据
@st.cache_data
def load_data():
    dev = pd.read_csv('dev.csv')
    vad = pd.read_csv('vad.csv')
    return dev, vad

dev, vad = load_data()

# 分类变量映射
CATEGORICAL_MAP = {
    'smoker': [1, 2, 3],
    'sex': [1, 2],
    'carace': [1, 2, 3, 4, 5],
    'drink': [1, 2],
    'sleep': [1, 2],
    'Hypertension': [1, 2],
    'Dyslipidemia': [1, 2]
}

# 特征顺序
feature_names = [
    'smoker', 'sex', 'carace', 'drink', 'sleep',
    'Hypertension', 'Dyslipidemia', 'HHR', 'RIDAGEYR',
    'INDFMPIR', 'BMXBMI', 'LBXWBCSI', 'LBXRBCSI'
]

# Streamlit界面
st.title("Co-occurrence Risk Predictor (R Model)")

# 模型性能展示
if st.sidebar.checkbox("Show Model Performance"):
    st.subheader("ROC Curve on Test Set")
    
    fpr, tpr, _ = roc_curve(test_probe['target'], test_probe['Yes'])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# 创建输入表单
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        smoker = st.selectbox("Smoker:", CATEGORICAL_MAP['smoker'], 
                            format_func=lambda x: ["Never", "Former", "Current"][x-1])
        sex = st.selectbox("Sex:", CATEGORICAL_MAP['sex'], 
                         format_func=lambda x: ["Female", "Male"][x-1])
        carace = st.selectbox("Race:", CATEGORICAL_MAP['carace'],
                            format_func=lambda x: ["Mexican","Hispanic","White","Black","Other"][x-1])
        drink = st.selectbox("Alcohol:", CATEGORICAL_MAP['drink'],
                           format_func=lambda x: ["No", "Yes"][x-1])
        sleep = st.selectbox("Sleep:", CATEGORICAL_MAP['sleep'],
                           format_func=lambda x: ["Problem", "Normal"][x-1])
        
    with col2:
        Hypertension = st.selectbox("Hypertension:", CATEGORICAL_MAP['Hypertension'],
                                  format_func=lambda x: ["No", "Yes"][x-1])
        Dyslipidemia = st.selectbox("Dyslipidemia:", CATEGORICAL_MAP['Dyslipidemia'],
                                  format_func=lambda x: ["No", "Yes"][x-1])
        HHR = st.number_input("HHR Ratio:", min_value=0.2, max_value=1.7, value=1.0)
        RIDAGEYR = st.number_input("Age:", min_value=20, max_value=80, value=50)
        INDFMPIR = st.number_input("Poverty Ratio:", min_value=0.0, max_value=5.0, value=2.0)
        BMXBMI = st.number_input("BMI:", min_value=15.0, max_value=60.0, value=25.0)
        LBXWBCSI = st.number_input("WBC:", min_value=2.0, max_value=20.0, value=6.0)
        LBXRBCSI = st.number_input("RBC:", min_value=2.5, max_value=7.0, value=4.0)

    submitted = st.form_submit_button("Predict")

def convert_to_r_factors(input_df):
    """将分类变量转换为R因子"""
    r_data = {}
    for col in input_df.columns:
        if col in CATEGORICAL_MAP:
            r_data[col] = robjects.FactorVector(
                input_df[col],
                levels=robjects.IntVector(CATEGORICAL_MAP[col]))
        else:
            r_data[col] = input_df[col].values
    return robjects.DataFrame(r_data)

# 预测函数
def r_predict(input_df):
    try:
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_data = convert_to_r_factors(input_df)
            r_pred = robjects.r['predict'](r_model, newdata=r_data, type="prob")
            pred_df = robjects.conversion.rpy2py(r_pred)
        return pred_df['Yes'].values
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

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
        predicted_class = 1 if prob_1 > 0.56 else 0
        
        # 显示结果
        st.success("### Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("High Risk Probability", f"{prob_1*100:.1f}%")
        with col2:
            st.metric("Recommended Action", 
                    "Immediate Intervention" if predicted_class else "Routine Monitoring")

        # 生成建议
        advice = """
        **Clinical Recommendations:**
        - Monthly cardiovascular screening
        - Daily blood pressure monitoring
        - Mediterranean diet plan
        - 150 mins/week moderate exercise
        {}"""
        st.info(advice.format("**▲ High Risk: Consult specialist immediately**" 
                            if predicted_class else "**▼ Low Risk: Maintain current lifestyle**"))

        # SHAP解释
        st.subheader("SHAP Feature Impact")
        with st.spinner('Generating explanation...'):
            background = shap.sample(vad[feature_names], 100)
            
            def shap_predict(data):
                return np.column_stack([1 - r_predict(pd.DataFrame(data, columns=feature_names)),
                                      r_predict(pd.DataFrame(data, columns=feature_names))])

            explainer = shap.KernelExplainer(shap_predict, background)
            shap_values = explainer.shap_values(input_df, nsamples=50)
            
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values[1], input_df, plot_type='bar', show=False)
            st.pyplot(fig)

        # LIME解释
        st.subheader("LIME Explanation")
        with st.spinner('Analyzing local features...'):
            def lime_predict(x):
                return np.column_stack([1 - r_predict(pd.DataFrame(x, columns=feature_names)),
                                      r_predict(pd.DataFrame(x, columns=feature_names))])

            explainer = LimeTabularExplainer(
                dev[feature_names].values,
                feature_names=feature_names,
                class_names=['Low Risk', 'High Risk'],
                mode='classification'
            )

            exp = explainer.explain_instance(
                input_df.values[0], 
                lime_predict,
                num_features=8
            )
            
            st.components.v1.html(exp.as_html(), height=800)

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.stop()
