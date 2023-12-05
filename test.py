import shap
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# 训练SVR模型
model = SVR()
model_prediction = model.fit(X_train, y_train)

# 创建SHAP的KernelExplainer对象
explainer = shap.KernelExplainer(model.predict, X_train)

# 计算SHAP值
shap_values = explainer.shap_values(X_test)

# 绘制SHAP值的摘要图
shap.summary_plot(shap_values, X_test, feature_names=boston.feature_names)
