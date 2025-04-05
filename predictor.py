import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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

# 预测函数
def r_predict(input_df, return_probs=False):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_data = robjects.conversion.py2rpy(input_df)
    
    r_pred = robjects.r['predict'](
        r_model, 
        newdata=r_data,
        type="prob"
    )
    
    with localconverter(robjects.default_converter + pandas2ri.converter):
        pred_df = robjects.conversion.rpy2py(r_pred)
    
    if return_probs:
        return pred_df['Yes'].values, pred_df['No'].values  # 返回阳性和阴性概率
    return pred_df['Yes'].values  # 默认只返回阳性概率

# Streamlit界面
st.title("Co-occurrence Risk Predictor (R Model)")

# 创建选项卡
tab1, tab2 = st.tabs(["Interactive Prediction", "Batch Evaluation"])

with tab1:
    # 交互式预测界面
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
            prob_1, prob_0 = r_predict(input_df, return_probs=True)
            prob_1 = prob_1[0]
            prob_0 = prob_0[0]
            predicted_class = 1 if prob_1 > 0.56 else 0
            
            # 显示结果
            st.success("### Prediction Results")
            st.metric("Comorbidity Risk", f"{prob_1*100:.1f}%", 
                    help="Probability of having both conditions")
            
            # 显示两类概率
            st.write(f"Class 1 (Positive) Probability: {prob_1:.4f}")
            st.write(f"Class 0 (Negative) Probability: {prob_0:.4f}")
            
            # 生成建议
            advice_template = """
            **Recommendations:**
            - Regular cardiovascular screening
            - Monitor blood pressure weekly
            - Mediterranean diet recommended
            {}"""
            st.info(advice_template.format("Immediate consultation needed!" if predicted_class==1 else "Maintain healthy lifestyle"))

            # SHAP解释
            st.subheader("Model Interpretation")
            
            # 准备解释数据
            background = shap.sample(vad[feature_names], 100)
            
            # 定义SHAP预测函数
            def shap_predict(data):
                input_df = pd.DataFrame(data, columns=feature_names)
                return np.column_stack([1-r_predict(input_df), r_predict(input_df)])
            
            # 创建解释器
            explainer = shap.KernelExplainer(shap_predict, background)
            shap_values = explainer.shap_values(input_df, nsamples=100)
            
            # 可视化
            st.subheader("Feature Impact")
            fig, ax = plt.subplots()
            shap.force_plot(explainer.expected_value[1], 
                           shap_values[0][:,1], 
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
            ).explain_instance(input_df.values[0], 
                             lambda x: np.column_stack((1-r_predict(pd.DataFrame(x, columns=feature_names)),
                                                       r_predict(pd.DataFrame(x, columns=feature_names)))))
            
            st.components.v1.html(lime_exp.as_html(), height=800)

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
            st.stop()

with tab2:
    # 批量评估界面
    st.header("Batch Prediction and Evaluation")
    
    # 上传测试数据
    uploaded_file = st.file_uploader("Upload test data (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            test_data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(test_data.head())
            
            if st.button("Run Batch Prediction"):
                # 确保有目标列
                if 'target' not in test_data.columns:
                    st.warning("No target column found in the data. Only predictions will be made.")
                    has_target = False
                else:
                    has_target = True
                    y_true = test_data['target'].values
                
                # 准备特征数据
                X_test = test_data[feature_names]
                
                # 执行批量预测
                with st.spinner("Running batch prediction..."):
                    try:
                        # 转换数据
                        with localconverter(robjects.default_converter + pandas2ri.converter):
                            r_X_test = robjects.conversion.py2rpy(X_test)
                        
                        # 获取概率预测
                        predict_r = robjects.r['predict']
                        r_pred_prob = predict_r(r_model, newdata=r_X_test, type="prob")
                        
                        # 转换回Python
                        with localconverter(robjects.default_converter + pandas2ri.converter):
                            pred_prob = robjects.conversion.rpy2py(r_pred_prob)
                        
                        # 获取预测结果
                        prob_yes = pred_prob['Yes'].values
                        prob_no = pred_prob['No'].values
                        y_pred = (prob_yes > 0.56).astype(int)
                        
                        # 显示结果
                        st.success("Batch prediction completed!")
                        
                        # 创建结果DataFrame
                        results_df = X_test.copy()
                        results_df['Predicted_Probability_1'] = prob_yes
                        results_df['Predicted_Probability_0'] = prob_no
                        results_df['Predicted_Class'] = y_pred
                        
                        if has_target:
                            results_df['True_Class'] = y_true
                        
                        st.dataframe(results_df)
                        
                        # 下载结果
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Predictions",
                            csv,
                            "batch_predictions.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        
                        # 如果有真实标签，显示评估指标
                        if has_target:
                            st.subheader("Model Performance Metrics")
                            
                            # 分类报告
                            st.write("Classification Report:")
                            report = classification_report(y_true, y_pred, output_dict=True)
                            st.table(pd.DataFrame(report).transpose())
                            
                            # 混淆矩阵
                            st.write("Confusion Matrix:")
                            cm = confusion_matrix(y_true, y_pred)
                            fig, ax = plt.subplots()
                            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                            disp.plot(ax=ax)
                            st.pyplot(fig)
                            
                            # 预测分布
                            st.write("Prediction Distribution:")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Predicted Classes:")
                                st.write(pd.Series(y_pred).value_counts().sort_index())
                            with col2:
                                st.write("True Classes:")
                                st.write(pd.Series(y_true).value_counts().sort_index())
                    
                    except Exception as e:
                        st.error(f"Batch prediction failed: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
