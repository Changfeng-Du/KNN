import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# 初始化R环境（必须最先执行）
pandas2ri.activate()
robjects.r['options'](warn=-1)

# 加载R模型（增加异常处理）
@st.cache_resource
def load_r_model():
    try:
        r_model = robjects.r['readRDS']('knn_model.rds')
        
        # 动态修补Pandas兼容性（参考代码1）
        if not hasattr(pd.DataFrame, 'iteritems'):
            pd.DataFrame.iteritems = pd.DataFrame.items
            
        return r_model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.stop()

r_model = load_r_model()

# 加载数据（添加路径检查）
@st.cache_data
def load_data():
    try:
        dev = pd.read_csv('dev.csv')
        vad = pd.read_csv('vad.csv')
        return dev, vad
    except FileNotFoundError:
        st.error("数据文件缺失！请检查 dev.csv 和 vad.csv 是否存在")
        st.stop()

dev, vad = load_data()

# 定义特征顺序（与模型输入严格一致）
feature_names = [
    'smoker', 'sex', 'carace', 'drink', 'sleep',
    'Hypertension', 'Dyslipidemia', 'HHR', 'RIDAGEYR',
    'INDFMPIR', 'BMXBMI', 'LBXWBCSI', 'LBXRBCSI'
]

# Streamlit界面
st.title("Co-occurrence Risk Predictor (R Model)")

# 创建输入表单（添加默认值验证）
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        smoker = st.selectbox("Smoker:", [1,2,3], index=0,
                            format_func=lambda x: ["Never", "Former", "Current"][x-1])
        sex = st.selectbox("Sex:", [1,2], index=0, 
                         format_func=lambda x: ["Female", "Male"][x-1])
        carace = st.selectbox("Race:", [1,2,3,4,5], index=0,
                            format_func=lambda x: ["Mexican","Hispanic","White","Black","Other"][x-1])
        drink = st.selectbox("Alcohol:", [1,2], index=0,
                           format_func=lambda x: ["No", "Yes"][x-1])
        sleep = st.selectbox("Sleep:", [1,2], index=0,
                           format_func=lambda x: ["Problem", "Normal"][x-1])
        
    with col2:
        Hypertension = st.selectbox("Hypertension:", [1,2], index=0,
                                  format_func=lambda x: ["No", "Yes"][x-1])
        Dyslipidemia = st.selectbox("Dyslipidemia:", [1,2], index=0,
                                  format_func=lambda x: ["No", "Yes"][x-1])
        HHR = st.number_input("HHR Ratio:", min_value=0.2, max_value=1.7, value=1.0, step=0.1)
        RIDAGEYR = st.number_input("Age:", min_value=20, max_value=80, value=50)
        INDFMPIR = st.number_input("Poverty Ratio:", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        BMXBMI = st.number_input("BMI:", min_value=15.0, max_value=60.0, value=25.0, step=0.1)
        LBXWBCSI = st.number_input("WBC:", min_value=2.0, max_value=20.0, value=6.0, step=0.1)
        LBXRBCSI = st.number_input("RBC:", min_value=2.5, max_value=7.0, value=4.0, step=0.1)

    submitted = st.form_submit_button("Predict")

# 预测函数（增加类型转换）
def r_predict(input_df):
    try:
        with localconverter(robjects.default_converter + pandas2ri.converter):
            # 确保输入为R兼容格式
            r_data = robjects.conversion.py2rpy(input_df.astype(float))
        
        r_pred = robjects.r['predict'](
            r_model, 
            newdata=r_data,
            type="prob"
        )
        
        with localconverter(robjects.default_converter + pandas2ri.converter):
            pred_df = robjects.conversion.rpy2py(r_pred)
            
        return pred_df['Yes'].values  # 返回阳性类概率
    except Exception as e:
        st.error(f"预测失败: {str(e)}")
        st.stop()

if submitted:
    # 构建输入数据（添加格式验证）
    try:
        input_data = [
            int(smoker), int(sex), int(carace), int(drink), int(sleep),
            int(Hypertension), int(Dyslipidemia), float(HHR), int(RIDAGEYR),
            float(INDFMPIR), float(BMXBMI), float(LBXWBCSI), float(LBXRBCSI)
        ]
        input_df = pd.DataFrame([input_data], columns=feature_names)
    except ValueError as e:
        st.error(f"输入格式错误: {str(e)}")
        st.stop()
    
    # 执行预测
    prob_1 = r_predict(input_df)[0]
    predicted_class = 1 if prob_1 > 0.56 else 0
    
    # 显示结果（优化显示逻辑）
    st.success("### Prediction Results")
    col_result1, col_result2 = st.columns(2)
    with col_result1:
        st.metric("Comorbidity Risk", 
                f"{prob_1*100:.1f}%", 
                help="Probability of having both conditions")
    with col_result2:
        st.metric("Risk Level", 
                "High Risk" if predicted_class == 1 else "Low Risk",
                delta_color="inverse")

    # 生成建议（优化建议内容）
    advice_template = """
    **Recommendations:**
    - 常规心血管筛查（每6个月）
    - 血压监测频率：{"每日" if predicted_class == 1 else "每周"}
    - 推荐饮食方案：{"地中海饮食" if predicted_class == 1 else "均衡饮食"}
    - 运动建议：{"每周5次低强度运动" if predicted_class == 1 else "每周3次中等强度运动"}
    """
    st.info(advice_template)

    # SHAP解释（修复可视化问题）
    st.subheader("特征影响分析")
    try:
        # 准备背景数据（与代码1一致）
        background = shap.sample(vad[feature_names], 100)
        
        # 定义SHAP预测函数（适配R模型）
        def shap_predict(data):
            input_df = pd.DataFrame(data, columns=feature_names)
            return np.column_stack([1 - r_predict(input_df), r_predict(input_df)])
        
        # 创建解释器（添加缓存）
        @st.cache_resource
        def create_explainer():
            return shap.KernelExplainer(
                model=shap_predict,
                data=background,
                link="identity"
            )
            
        explainer = create_explainer()
        shap_values = explainer.shap_values(input_df)
        
        # 可视化（调整显示参数）
        fig, ax = plt.subplots()
        shap.force_plot(
            base_value=explainer.expected_value[1],
            shap_values=shap_values[0][:,1],
            features=input_df.iloc[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            text_rotation=15
        )
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"SHAP解释失败: {str(e)}")

    # LIME解释（修复函数参数）
    st.subheader("局部可解释性分析")
    try:
        lime_explainer = LimeTabularExplainer(
            training_data=background.values,
            feature_names=feature_names,
            class_names=['Low Risk','High Risk'],
            mode='classification',
            discretize_continuous=False
        )
        
        exp = lime_explainer.explain_instance(
            data_row=input_df.values[0],
            predict_fn=lambda x: np.column_stack([
                1 - r_predict(pd.DataFrame(x, columns=feature_names)),
                r_predict(pd.DataFrame(x, columns=feature_names))
            ]),
            num_features=10
        )
        
        st.components.v1.html(exp.as_html(), height=800)
    except Exception as e:
        st.error(f"LIME解释失败: {str(e)}")
