import streamlit as st
import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

# 配置页面
st.set_page_config(page_title="Medical Risk Predictor", layout="wide")

# --------------------- 初始化R环境 ---------------------
try:
    pandas2ri.activate()
    
    # 安装必要R包
    robjects.r('''
    if (!require("caret")) {
        install.packages("caret", repos="http://cran.r-project.org", quiet=TRUE)
        library(caret, quietly=TRUE)
    }
    if (!require("pROC")) {
        install.packages("pROC", repos="http://cran.r-project.org", quiet=TRUE)
        library(pROC, quietly=TRUE)
    }
    ''')

    # 定义R预测函数
    robjects.r('''
    safe_predict <- function(model, newdata) {
        tryCatch({
            # 强制转换分类变量
            factor_cols <- c("smoker", "sex", "carace", "drink", "sleep",
                           "Hypertension", "Dyslipidemia")
            newdata[factor_cols] <- lapply(newdata[factor_cols], factor)
            
            # 确保因子水平与训练时一致
            for(col in factor_cols) {
                levels(newdata[[col]]) <- model$trainingData[[col]] %>% levels()
            }
            
            # 生成预测
            pred <- predict(model, newdata = newdata, type = "prob")
            if ("Yes" %in% colnames(pred)) return(pred$Yes) else return(pred[,2])
        }, error = function(e) {
            stop(paste("Prediction failed:", e$message))
        })
    }
    ''')
except Exception as e:
    st.error(f"R环境初始化失败: {str(e)}")
    st.stop()

# --------------------- 数据加载 ---------------------
@st.cache_data
def load_data():
    # 加载预测概率
    train_probe = pd.read_csv('train_probe.csv')
    test_probe = pd.read_csv('test_probe.csv')
    
    # 加载原始数据
    dev = pd.read_csv('dev.csv')
    vad = pd.read_csv('vad.csv')
    
    return train_probe, test_probe, dev, vad

try:
    train_probe, test_probe, dev, vad = load_data()
    st.sidebar.success("数据加载完成!")
except Exception as e:
    st.error(f"数据加载失败: {str(e)}")
    st.stop()

# --------------------- 模型加载 ---------------------
@st.cache_resource
def load_model():
    try:
        if not os.path.exists('knn_model.rds'):
            raise FileNotFoundError("模型文件 knn_model.rds 不存在")
            
        r_model = robjects.r['readRDS']('knn_model.rds')
        
        # 验证模型结构
        if not all(robjects.r['names'](r_model).contains(['method', 'trainingData'])):
            raise ValueError("无效的模型结构")
            
        return r_model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.stop()

r_model = load_model()

# --------------------- 特征配置 ---------------------
feature_config = {
    'smoker': {'label': '吸烟状态', 'options': {1: '从不', 2: '曾经', 3: '当前'}},
    'sex': {'label': '性别', 'options': {1: '女', 2: '男'}},
    'carace': {'label': '种族', 'options': {1: '墨西哥裔', 2: '西班牙裔', 3: '白人', 4: '黑人', 5: '其他'}},
    'drink': {'label': '饮酒', 'options': {1: '否', 2: '是'}},
    'sleep': {'label': '睡眠质量', 'options': {1: '有问题', 2: '正常'}},
    'Hypertension': {'label': '高血压', 'options': {1: '无', 2: '有'}},
    'Dyslipidemia': {'label': '血脂异常', 'options': {1: '无', 2: '有'}},
    'HHR': {'label': '心率变异比', 'type': 'number', 'min': 0.5, 'max': 2.0, 'step': 0.1},
    'RIDAGEYR': {'label': '年龄', 'type': 'number', 'min': 20, 'max': 80},
    'INDFMPIR': {'label': '贫困指数', 'type': 'number', 'min': 0.0, 'max': 5.0, 'step': 0.1},
    'BMXBMI': {'label': 'BMI', 'type': 'number', 'min': 15.0, 'max': 50.0, 'step': 0.1},
    'LBXWBCSI': {'label': '白细胞计数', 'type': 'number', 'min': 2.0, 'max': 20.0, 'step': 0.1},
    'LBXRBCSI': {'label': '红细胞计数', 'type': 'number', 'min': 2.5, 'max': 7.0, 'step': 0.1}
}

# --------------------- 用户输入界面 ---------------------
st.title("医疗共病风险预测系统")
with st.expander("患者信息输入", expanded=True):
    input_data = {}
    cols = st.columns(3)
    
    # 分类特征
    categorical_features = ['smoker', 'sex', 'carace', 'drink', 'sleep', 'Hypertension', 'Dyslipidemia']
    for i, feat in enumerate(categorical_features):
        with cols[i%3]:
            config = feature_config[feat]
            input_data[feat] = st.selectbox(
                config['label'],
                options=list(config['options'].keys()),
                format_func=lambda x, f=feat: feature_config[f]['options'][x]
            )
    
    # 数值特征
    numerical_features = ['HHR', 'RIDAGEYR', 'INDFMPIR', 'BMXBMI', 'LBXWBCSI', 'LBXRBCSI']
    for i, feat in enumerate(numerical_features):
        with cols[i%3]:
            config = feature_config[feat]
            input_data[feat] = st.number_input(
                config['label'],
                min_value=config['min'],
                max_value=config['max'],
                value=(config['min'] + config['max'])/2,
                step=config.get('step', 1.0)
            )

# --------------------- 预测执行 ---------------------
if st.button("开始预测"):
    try:
        # 转换输入数据
        input_df = pd.DataFrame([input_data])
        
        # R预测
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_input = robjects.conversion.py2rpy(input_df)
        
        prob = robjects.globalenv['safe_predict'](r_model, r_input)[0]
        
        # 显示结果
        st.success(f"预测结果：共病风险为 {prob*100:.1f}%")
        
        # 决策阈值
        threshold = 0.56
        decision = "高风险" if prob >= threshold else "低风险"
        color = "red" if prob >= threshold else "green"
        st.markdown(f"<h3 style='text-align: center; color: {color};'>{decision}</h3>", 
                   unsafe_allow_html=True)
        
        # --------------------- 模型解释 ---------------------
        st.subheader("模型解释")
        
        # SHAP解释
        with st.spinner('生成SHAP解释...'):
            explainer = shap.KernelExplainer(
                lambda x: np.array([robjects.globalenv['safe_predict'](r_model, 
                                    robjects.conversion.py2rpy(pd.DataFrame(x, columns=input_df.columns))]),
                shap.sample(vad[input_df.columns], 100)
            )
            shap_values = explainer.shap_values(input_df)
            
            fig, ax = plt.subplots()
            shap.force_plot(explainer.expected_value, 
                           shap_values[0], 
                           input_df.iloc[0],
                           matplotlib=True, show=False)
            st.pyplot(fig)
        
        # LIME解释
        with st.spinner('生成LIME解释...'):
            lime_exp = LimeTabularExplainer(
                dev[input_df.columns].values,
                feature_names=input_df.columns,
                class_names=['低风险', '高风险'],
                mode='classification'
            ).explain_instance(
                input_df.values[0], 
                lambda x: np.array([robjects.globalenv['safe_predict'](r_model, 
                                  robjects.conversion.py2rpy(pd.DataFrame(x, columns=input_df.columns))])
            
            st.components.v1.html(lime_exp.as_html(), height=800)
        
        # --------------------- 性能可视化 ---------------------
        st.subheader("模型性能")
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(test_probe.target, test_probe.Yes)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('假阳性率')
        ax.set_ylabel('真阳性率')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        
        # 校准曲线
        prob_true, prob_pred = calibration_curve(test_probe.target, test_probe.Yes, n_bins=10)
        
        fig, ax = plt.subplots()
        ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='模型')
        ax.plot([0, 1], [0, 1], linestyle='--', label='理想校准')
        ax.set_xlabel('预测概率')
        ax.set_ylabel('实际概率')
        ax.legend()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"预测失败: {str(e)}")
        st.stop()

# --------------------- 侧边栏分析 ---------------------
with st.sidebar:
    st.header("模型分析")
    
    # 特征重要性
    if st.checkbox("显示特征重要性"):
        try:
            # 获取R模型的变量重要性
            with localconverter(robjects.default_converter + pandas2ri.converter):
                var_imp = robjects.r('varImp')(r_model)
                var_imp_df = robjects.conversion.rpy2py(var_imp)
            
            st.bar_chart(var_imp_df.set_index('Feature')['Importance'])
        except:
            st.warning("无法获取变量重要性")
    
    # 概率分布
    if st.checkbox("显示概率分布"):
        fig, ax = plt.subplots()
        ax.hist(test_probe[test_probe.target==0].Yes, bins=30, alpha=0.5, label='低风险')
        ax.hist(test_probe[test_probe.target==1].Yes, bins=30, alpha=0.5, label='高风险')
        ax.set_xlabel('预测概率')
        ax.set_ylabel('频数')
        ax.legend()
        st.pyplot(fig)

# --------------------- 环境信息 ---------------------
with st.sidebar.expander("系统信息"):
    st.write(f"R版本: {robjects.r('version$version.string')[0]}")
    st.write(f"Python版本: {sys.version}")
    st.write("工作目录:", os.getcwd())
