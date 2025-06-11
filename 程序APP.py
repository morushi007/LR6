import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib
import shap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import os

# 设置页面配置
st.set_page_config(
    page_title="PCNL Post-Operative Fever Prediction Model",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加CSS样式
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
    }
    h1 {
        color: #2C3E50;
    }
    h2 {
        color: #3498DB;
    }
    .footer {
        margin-top: 2rem;
        text-align: center;
        color: #7F8C8D;
    }
</style>
""", unsafe_allow_html=True)

# 显示标题与说明
st.title("PCNL Post-Operative Fever Prediction Model")
st.markdown("### A machine learning-based tool to predict post-operative fever risk after percutaneous nephrolithotomy")

# 创建侧边栏信息
with st.sidebar:
    st.header("About this Model")
    st.info(
        """
        This prediction model is based on clinical features to assess the risk of fever after 
        Percutaneous Nephrolithotomy (PCNL).
        
        Please enter the patient's information on the right to obtain the prediction results.
        """
    )
    st.header("Feature Description")
    st.markdown("""
    - **NLR**: Neutrophil-to-Lymphocyte Ratio
    - **LMR**: Lymphocyte-to-Monocyte Ratio
    - **BMI**: Body Mass Index
    - **Mayo Score**: Mayo Surgical Complexity Score
    - **Channel Size**: Size of the nephroscope channel
    """)

# 确保模型文件存在
def load_model():
    try:
        # 加载模型信息字典
        model_info = joblib.load('LR.pkl')
        return model_info
    except:
        # 如果加载失败，显示错误信息
        st.error("Model file 'LR.pkl' not found. Please ensure the model file is uploaded to the same directory as the application.")
        return None

# 定义特征范围 - 使用您提供的新特征范围
feature_ranges = {
    "MayoScore_bin": {"type": "categorical", "options": ["<3", "≥3"], "default": "<3", "description": "Mayo Score"},
    "Diabetes_mellitus": {"type": "categorical", "options": ["No", "Yes"], "default": "No", "description": "Diabetes mellitus"},
    "UrineLeuk_bin": {"type": "categorical", "options": ["=0", ">0"], "default": "=0", "description": "Urine Leukocytes"},
    "Sex": {"type": "categorical", "options": ["Male", "Female"], "default": "Male", "description": "Sex"},
    "BMI": {"type": "numerical", "min": 10.0, "max": 50.0, "default": 24.0, "description": "Body Mass Index (kg/m²)"},
    "Preoperative_urinary_nitrite": {"type": "categorical", "options": ["=0", ">0"], "default": "=0", "description": "Preoperative Urinary Nitrite"},
    "NLR": {"type": "numerical", "min": 0.0, "max": 20.0, "default": 3.0, "description": "Neutrophil-to-Lymphocyte Ratio (NLR)"},
    "LMR": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 5.0, "description": "Lymphocyte-to-Monocyte Ratio (LMR)"},
    "Channel_size": {"type": "categorical", "options": ["18F", "20F"], "default": "18F", "description": "Channel Size"},
    "Preoperative_L": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 1.8, "description": "Preoperative Lymphocyte Count (×10^9/L)"},
    "Preoperative_M": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 0.6, "description": "Preoperative Monocyte Count (×10^9/L)"}
}

# 创建用户输入页面布局
st.header("Enter Patient Information")

# 使用列布局改善用户界面
col1, col2, col3 = st.columns(3)

# 创建空字典存储特征值
input_features = {}

# 将特征分配到列中
feature_columns = {
    0: col1,
    1: col2,
    2: col3
}

# 将特征分组到列中
i = 0
for feature, properties in feature_ranges.items():
    col = feature_columns[i % 3]
    with col:
        if properties["type"] == "numerical":
            input_features[feature] = st.number_input(
                label=f"{properties['description']} ({feature})",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                help=f"Range: {properties['min']} - {properties['max']}"
            )
        elif properties["type"] == "categorical":
            input_features[feature] = st.selectbox(
                label=f"{properties['description']} ({feature})",
                options=properties["options"],
                index=properties["options"].index(properties["default"]),
                help=f"Select an option"
            )
    i += 1

# 添加分隔线
st.markdown("---")

# 添加预测按钮
predict_button = st.button("Predict Fever Risk", use_container_width=True)

# 当按钮被点击时进行预测
if predict_button:
    # 加载模型
    model_info = load_model()
    
    if model_info:
        # 获取分类特征和数值特征列表
        categorical_features = [f for f, p in feature_ranges.items() if p["type"] == "categorical"]
        numerical_features = [f for f, p in feature_ranges.items() if p["type"] == "numerical"]
        
        # 准备数据框
        input_df = pd.DataFrame([input_features])
        
        # 处理分类特征 - 根据模型训练方式调整
        for feature in categorical_features:
            # 根据新的特征范围调整编码方式
            if feature == "MayoScore_bin":
                input_df[feature] = 1 if input_features[feature] == "≥3" else 0
            elif feature == "Diabetes_mellitus":
                input_df[feature] = 1 if input_features[feature] == "Yes" else 0
            elif feature == "UrineLeuk_bin":
                input_df[feature] = 1 if input_features[feature] == ">0" else 0
            elif feature == "Sex":
                input_df[feature] = 1 if input_features[feature] == "Male" else 0
            elif feature == "Preoperative_urinary_nitrite":
                input_df[feature] = 1 if input_features[feature] == ">0" else 0
            elif feature == "Channel_size":
                input_df[feature] = 1 if input_features[feature] == "18F" else 0
        
        try:
            # 标准化数值特征
            X_scaled = input_df.copy()
            X_scaled[model_info['numerical_features']] = model_info['scaler'].transform(
                input_df[model_info['numerical_features']]
            )
            
            # 进行预测
            predicted_proba = model_info['lr_model'].predict_proba(X_scaled)[0]
            # 假设模型是二分类，1表示发热
            fever_probability = predicted_proba[1] * 100
            
            # 显示结果
            st.markdown("## Prediction Results")
            
            # 创建结果显示区域
            result_col1, result_col2 = st.columns([2, 1])
            
            with result_col1:
                # 根据概率值显示不同的风险级别
                if fever_probability < 25:
                    risk_level = "Low Risk"
                    color = "green"
                elif fever_probability < 50:
                    risk_level = "Moderate-Low Risk"
                    color = "lightgreen"
                elif fever_probability < 75:
                    risk_level = "Moderate-High Risk"
                    color = "orange"
                else:
                    risk_level = "High Risk"
                    color = "red"
                
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {color}; text-align: center;">
                    <h2 style="color: white;">Post-operative Fever Risk: {risk_level}</h2>
                    <h3 style="color: white;">Predicted Probability: {fever_probability:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # 添加结果解释
                st.markdown(f"""
                ### Result Interpretation
                - The predicted probability of post-operative fever for this patient is **{fever_probability:.2f}%**
                - Risk Level: **{risk_level}**
                
                **Note**: This prediction is for clinical reference only and should not replace professional medical judgment.
                """)
            
            with result_col2:
                # 创建简单的概率可视化
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie([fever_probability, 100-fever_probability], 
                       labels=["Fever Risk", "No Fever Risk"],
                       colors=[color, "lightgrey"],
                       autopct='%1.1f%%',
                       startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            
            # 特征重要性分析
            st.markdown("## Feature Impact Analysis")
            st.info("The chart below shows how each feature influences the prediction. Features with positive values (red) increase fever risk, while features with negative values (blue) decrease risk.")
            
            try:
                # 确认模型是否有系数属性（如逻辑回归模型）
                if hasattr(model_info['lr_model'], 'coef_'):
                    # 创建特征系数的DataFrame
                    coef_df = pd.DataFrame({
                        'Feature': input_df.columns.tolist(),
                        'Coefficient': model_info['lr_model'].coef_[0]
                    })
                    
                    # 按照系数绝对值排序，以便展示最重要的特征
                    sorted_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
                    
                    # 创建系数条形图
                    fig, ax = plt.subplots(figsize=(10, 8))
                    colors = ['#3498db' if c < 0 else '#e74c3c' for c in sorted_df['Coefficient']]
                    
                    # 绘制水平条形图
                    bars = ax.barh(sorted_df['Feature'], sorted_df['Coefficient'], color=colors)
                    
                    # 添加垂直线表示零点
                    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                    
                    # 设置坐标轴标签和标题
                    ax.set_xlabel('Impact on Fever Risk', fontsize=12)
                    ax.set_title('Feature Impact on Post-operative Fever Prediction', fontsize=14)
                    
                    # 为每个条添加数值标签
                    for bar in bars:
                        width = bar.get_width()
                        label_x_pos = width + 0.01 if width > 0 else width - 0.01
                        label_ha = 'left' if width > 0 else 'right'
                        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                                f'{width:.3f}', va='center', ha=label_ha, fontsize=10)
                    
                    # 添加图例
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='#e74c3c', label='Increases Fever Risk'),
                        Patch(facecolor='#3498db', label='Decreases Fever Risk')
                    ]
                    ax.legend(handles=legend_elements, loc='lower right')
                    
                    # 调整布局并显示
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # 添加特征重要性表格，按照系数绝对值排序
                    st.subheader("Feature Importance Table")
                    importance_df = coef_df.copy()
                    importance_df['Absolute Impact'] = np.abs(importance_df['Coefficient'])
                    importance_df = importance_df.sort_values('Absolute Impact', ascending=False)
                    importance_df['Direction'] = importance_df['Coefficient'].apply(
                        lambda x: "Increases Risk" if x > 0 else "Decreases Risk")
                    
                    # 显示表格
                    st.table(importance_df[['Feature', 'Coefficient', 'Direction', 'Absolute Impact']])
                    
                    # 解释患者的具体风险因素
                    st.subheader("Patient-Specific Risk Factors")
                    
                    # 获取对该患者影响最大的正向和负向因素
                    top_positive = sorted_df[sorted_df['Coefficient'] > 0].head(3)
                    top_negative = sorted_df[sorted_df['Coefficient'] < 0].head(3)
                    
                    if not top_positive.empty:
                        st.markdown("#### Top factors increasing this patient's fever risk:")
                        for idx, row in top_positive.iterrows():
                            feature_name = row['Feature']
                            feature_desc = next((p["description"] for f, p in feature_ranges.items() if f == feature_name), feature_name)
                            st.markdown(f"- **{feature_desc}**: Coefficient = {row['Coefficient']:.3f}")
                    
                    if not top_negative.empty:
                        st.markdown("#### Top factors decreasing this patient's fever risk:")
                        for idx, row in top_negative.iterrows():
                            feature_name = row['Feature']
                            feature_desc = next((p["description"] for f, p in feature_ranges.items() if f == feature_name), feature_name)
                            st.markdown(f"- **{feature_desc}**: Coefficient = {row['Coefficient']:.3f}")
                    
                else:
                    st.info("Feature impact analysis is not available for this model type.")
                    
            except Exception as e:
                st.error(f"Unable to generate feature impact visualization: {str(e)}")
                st.markdown("""
                Feature impact analysis is currently unavailable. This may be due to:
                1. The model structure is not compatible with coefficient-based analysis
                2. Required visualization libraries are missing
                3. The model file format is not as expected
                
                Please ensure the model is a coefficients-based model like Logistic Regression.
                """)
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.markdown("""
            Possible reasons:
            1. Input data format does not match model expectations
            2. Model file may be corrupted or incompatible
            """)

# 添加页脚
st.markdown("""
<div class="footer">
    <p>© 2025 PCNL Post-Operative Fever Prediction Model | This tool is for clinical reference only and should not replace professional medical judgment</p>
</div>
""", unsafe_allow_html=True)

# 添加"如何使用"折叠面板
with st.expander("How to Use This Tool"):
    st.markdown("""
    1. Enter the patient's clinical parameters in the form above
    2. Click the "Predict Fever Risk" button
    3. Review the prediction results and feature impact analysis
    4. Use the results as a reference for clinical decision-making
    
    **Notes**:
    - All values must be within the specified ranges
    - For missing data, it's recommended to use common clinical default values
    - This model is trained on historical data and may not apply to all clinical scenarios
    
    **Feature Descriptions**:
    - **NLR**: Neutrophil-to-Lymphocyte Ratio (calculated as neutrophil count / lymphocyte count)
    - **LMR**: Lymphocyte-to-Monocyte Ratio (calculated as lymphocyte count / monocyte count)
    - **BMI**: Body Mass Index (calculated as weight in kg / height in m²)
    - **Mayo Score**: Mayo Surgical Complexity Score (<3 = low complexity, ≥3 = high complexity)
    - **Channel Size**: Size of the nephroscope channel (18F or 20F)
    - **Urinary Nitrite**: Preoperative urinary nitrite test result (=0 = negative, >0 = positive)
    - **Urine Leukocytes**: Urine leukocyte test result (=0 = negative, >0 = positive)
    """)