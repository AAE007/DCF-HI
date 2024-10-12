import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 读取数据
data_path = r'../data/CMAPSS/train_FD004.txt'
column_names = ['unit_number', 'time_in_cycles'] + [f'operational_setting_{i}' for i in range(1, 4)] + [f'sensor_measurement_{i}' for i in range(1, 22)]
data = pd.read_csv(data_path, sep='\s+', header=None, names=column_names, engine='python')

# 检查并删除空白列
data = data.dropna(axis=1, how='all')

# 数据预处理
# 归一化
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data.iloc[:, 2:]), columns=data.columns[2:])

# 健康指标构建
# 使用主成分分析 (PCA) 提取特征并构建健康指标
pca = PCA(n_components=1)
data['health_indicator'] = pca.fit_transform(data_scaled)

# 绘制健康指标随时间变化的图像
for unit in data['unit_number'].unique():
    plt.plot(data[data['unit_number'] == unit]['time_in_cycles'], data[data['unit_number'] == unit]['health_indicator'], label=f'Unit {unit}')
plt.xlabel('Time in Cycles')
plt.ylabel('Health Indicator')
plt.title('Health Indicator over Time for Different Units')
plt.legend()
plt.show()

# 计算和展示健康指标随时间的变化趋势
trend = data.groupby('time_in_cycles')['health_indicator'].mean()
plt.plot(trend.index, trend.values)
plt.xlabel('Time in Cycles')
plt.ylabel('Average Health Indicator')
plt.title('Average Health Indicator over Time')
plt.show()
