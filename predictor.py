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

# 分类变量映射
CATEGORICAL_MAP = {
    'smoker': {
        'values': [1, 2, 3],
        'labels': ["Never", "Former", "Current"]
    },
    'sex': {
        'values': [1, 2],
        'labels': ["Female", "Male"]
    },
    'carace': {
        'values': [1, 2, 3, 4, 5],
        'labels': ["Mexican", "Hispanic", "White", "Black", "Other"]
    },
    'drink': {
        'values': [1, 2],
        'labels': ["No", "Yes"]
    },
    'sleep': {
        'values': [1, 2],
        'labels': ["Problem", "Normal"]
    },
    'Hypertension': {
        'values': [1, 2],
        'labels': ["No", "Yes"]
    },
    'Dyslipidemia': {
        'values': [1, 2],
        'labels': ["No", "Yes"]
    }
}

# 特征顺序
feature_names = [
    'smoker', 'sex', 'carace', 'drink', 'sleep',
    'Hypertension', 'Dyslipidemia', 'HHR', 'RIDAGEYR',
    'INDFMPIR', 'BMXBMI', 'LBXWBCSI', 'LBXRBCSI'
]

# 加载资源和数据
@st.cache_resource
def load_resources():
    # 加载R模型
    r_model = robjects.r['readRDS']('knn_model.rds')
    
    # 动态修补Pandas兼容性
    if not hasattr(pd.DataFrame, 'iteritems'):
        pd.DataFrame.iteritems = pd.DataFrame.items
        
    return r_model

@st.cache_data
def load_data():
    dev = pd.read_csv('dev.csv')
    vad = pd.read_csv('vad.csv')
    return dev, vad

r_model = load_resources()
dev, vad = load_data()

# 数据转换函数
def convert_to_r_factors(input_df):
    """将分类变量转换为R因子，保留原始数值类型"""
    r_data = {}
    for col in input_df.columns:
        if col in CATEGORICAL_MAP:
            # 转换为字符串类型因子
            factor_levels = [str(x) for x in CATEGORICAL_MAP[col]['values']]
            r_data[col] = robjects.FactorVector(
                input_df[col].astype(str),  # 转换为字符串
                levels=robjects.StrVector(factor_levels))
        else:
            r_data[col] = input_df[col].values
    return robjects.DataFrame(r_data)

# 预测函数
def r_predict(input_df):
    try:
        # 类型强制转换
        type_map = {
            col: 'int64' if col in CATEGORICAL_MAP else 'float64'
            for col in feature_names
        }
        input_df = input_df.astype(type_map)
        
        # R数据转换
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_data = convert_to_r_factors(input_df)
            r_pred = robjects.r['predict'](
                r_model, 
                newdata=r_data,
                type="prob"
            )
            with localconverter(robjects.default_converter + pandas2ri.converter):
                pred_df = robjects.conversion.rpy2py(r_pred)
        
        # 确保返回概率值
        if isinstance(pred_df, pd.DataFrame):
            return pred_df['Yes'].values
        elif isinstance(pred_df, np.ndarray):
            return pred_df[:, 1] if pred_df.shape[1] > 1 else pred_df.flatten()
        else:
            raise ValueError("Unexpected prediction output format")
            
    except Exception as e:
        st.error(f"Prediction Error Details:\n{str(e)}")
        raise RuntimeError("Prediction failed due to data processing error")

# Streamlit界面
st.title("Co-occurrence Risk Predictor (R Model)")

# 创建输入表单
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        smoker = st.selectbox("Smoker:", 
                            CATEGORICAL_MAP['smoker']['values'],
                            format_func=lambda x: CATEGORICAL_MAP['smoker']['labels'][x-1])
        sex = st.selectbox("Sex:", 
                         CATEGORICAL_MAP['sex']['values'],
                         format_func=lambda x: CATEGORICAL_MAP['sex']['labels'][x-1])
        carace = st.selectbox("Race:",
                            CATEGORICAL_MAP['carace']['values'],
                            format_func=lambda x: CATEGORICAL_MAP['carace']['labels'][x-1])
        drink = st.selectbox("Alcohol:",
                           CATEGORICAL_MAP['drink']['values'],
                           format_func=lambda x: CATEGORICAL_MAP['drink']['labels'][x-1])
        sleep = st.selectbox("Sleep:",
                           CATEGORICAL_MAP['sleep']['values'],
                           format_func=lambda x: CATEGORICAL_MAP['sleep']['labels'][x-1])
        
    with col2:
        Hypertension = st.selectbox("Hypertension:",
                                  CATEGORICAL_MAP['Hypertension']['values'],
                                  format_func=lambda x: CATEGORICAL_MAP['Hypertension']['labels'][x-1])
        Dyslipidemia = st.selectbox("Dyslipidemia:",
                                  CATEGORICAL_MAP['Dyslipidemia']['values'],
                                  format_func=lambda x: CATEGORICAL_MAP['Dyslipidemia']['labels'][x-1])
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
        int(smoker), int(sex), int(carace), int(drink), int(sleep),
        int(Hypertension), int(Dyslipidemia), float(HHR), int(RIDAGEYR),
        float(INDFMPIR), float(BMXBMI), float(LBXWBCSI), float(LBXRBCSI)
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
                return np.column_stack([
                    1 - r_predict(pd.DataFrame(data, columns=feature_names)),
                    r_predict(pd.DataFrame(data, columns=feature_names))
                ])

            explainer = shap.KernelExplainer(shap_predict, background)
            shap_values = explainer.shap_values(input_df, nsamples=50)
            
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values[1], input_df, plot_type='bar', show=False)
            st.pyplot(fig)

        # LIME解释
        st.subheader("LIME Explanation")
        with st.spinner('Analyzing local features...'):
            def lime_predict(x):
                return np.column_stack([
                    1 - r_predict(pd.DataFrame(x, columns=feature_names)),
                    r_predict(pd.DataFrame(x, columns=feature_names))
                ])

            lime_explainer = LimeTabularExplainer(
                dev[feature_names].values,
                feature_names=feature_names,
                class_names=['Low Risk', 'High Risk'],
                mode='classification'
            )

            exp = lime_explainer.explain_instance(
                input_df.values[0], 
                lime_predict,
                num_features=8
            )
            
            st.components.v1.html(exp.as_html(), height=800)

    except Exception as e:
        st.error(f"System Error: {str(e)}")
        st.stop()
