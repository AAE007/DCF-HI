import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import matplotlib.pyplot as plt

# 读取训练集数据
train_data = pd.read_excel('../data/CMAPSS/train_data_FD003.xlsx')

# 选择需要的传感器列
sensor_cols = ['sensor_measurement_2', 'sensor_measurement_3', 'sensor_measurement_4',
               'sensor_measurement_7', 'sensor_measurement_8', 'sensor_measurement_9',
               'sensor_measurement_11', 'sensor_measurement_12', 'sensor_measurement_13',
               'sensor_measurement_14', 'sensor_measurement_15', 'sensor_measurement_17',
               'sensor_measurement_20', 'sensor_measurement_21']

# 初始化MinMaxScaler用于正则化
scaler = MinMaxScaler(feature_range=(-1, 1))

# 存储每个引擎的健康指标
health_indices = []

# 获取所有的引擎编号
engine_ids = train_data['unit_number'].unique()[5:6]

# 对每个引擎逐个处理
for engine_id in engine_ids:
    # 过滤出当前引擎的数据
    engine_data = train_data[train_data['unit_number'] == engine_id]

    # 只保留需要的传感器数据
    sensor_data = engine_data[sensor_cols]

    # 对传感器数据进行正则化
    normalized_data = scaler.fit_transform(sensor_data)

    # 滑窗处理
    window_size = 30
    num_windows = len(normalized_data) - window_size + 1

    engine_health_index = []

    # 获取第一个滑窗
    window_1 = normalized_data[0:window_size]

    for i in range(1, num_windows):
        window_2 = normalized_data[i:i + window_size]

        # 对每个传感器数据进行贝叶斯高斯分布拟合
        diff_areas = []
        for sensor_index in range(window_1.shape[1]):
            mu1, std1 = norm.fit(window_1[:, sensor_index])
            mu2, std2 = norm.fit(window_2[:, sensor_index])

            # 计算两个分布的重叠面积
            diff_mu = np.abs(mu1 - mu2)
            combined_sigma = np.sqrt(std1 ** 2 + std2 ** 2)
            z = diff_mu / combined_sigma * np.sqrt(2)
            overlap_area = 2 * norm.cdf(-z / 2)
            area_diff = overlap_area * overlap_area
            diff_areas.append(area_diff)

        # 计算14个传感器分布差的均值
        mean_diff_area = np.mean(diff_areas)

        # 健康指标为1-面积的差值的均值
        health_index = mean_diff_area
        engine_health_index.append(health_index)

    health_indices.append(engine_health_index)

# 将第一个引擎的健康指标画图
plt.plot(health_indices[0])
plt.xlabel('Cycle')
plt.ylabel('Health Index')
plt.title('Health Index of Engine 1')
plt.show()
