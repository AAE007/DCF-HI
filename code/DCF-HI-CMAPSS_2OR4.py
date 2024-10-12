import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import matplotlib.pyplot as plt

# 读取训练集数据
train_data = pd.read_excel('../data/CMAPSS/train_data_FD002.xlsx')

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
engine_ids = train_data['unit_number'].unique()[111:112]

# 定义操作条件阈值及其范围
op_conditions = [0, 10, 20, 25, 35, 42]
condition_ranges = [(condition - 3, condition + 3) for condition in op_conditions]

# 对每个引擎逐个处理
for engine_id in engine_ids:
    # 过滤出当前引擎的数据
    engine_data = train_data[train_data['unit_number'] == engine_id]

    # 根据操作条件分组
    grouped_data = {}
    for condition, (lower_bound, upper_bound) in zip(op_conditions, condition_ranges):
        condition_data = engine_data[(engine_data['operational_setting_1'] >= lower_bound) &
                                     (engine_data['operational_setting_1'] <= upper_bound)]
        if not condition_data.empty:
            # 只保留需要的传感器数据
            condition_sensor_data = condition_data[sensor_cols]
            # 对传感器数据进行列正则化
            normalized_condition_data = scaler.fit_transform(condition_sensor_data)
            grouped_data[condition] = normalized_condition_data

    engine_health_index = []

    # 初始化滑窗索引
    window_indices = {condition: 0 for condition in grouped_data.keys()}
    initial_windows = {condition: data[0:5] for condition, data in grouped_data.items()}

    # 循环滑窗，直到所有条件数据滑动完
    while any(window_indices[condition] + 5 <= len(data) for condition, data in grouped_data.items()):
        for condition, data in grouped_data.items():
            if window_indices[condition] + 5 > len(data):
                continue

            window_1 = initial_windows[condition]
            window_2 = data[window_indices[condition]:window_indices[condition] + 5]

            # 对每个传感器数据进行贝叶斯高斯分布拟合
            diff_areas = []
            for sensor_index in range(window_1.shape[1]):
                mu1, std1 = norm.fit(window_1[:, sensor_index])
                mu2, std2 = norm.fit(window_2[:, sensor_index])

                # 计算两个分布的面积差值
                # 计算两个分布的重叠面积
                diff_mu = np.abs(mu1 - mu2)
                combined_sigma = np.sqrt(std1 ** 2 + std2 ** 2)
                z = diff_mu / combined_sigma * np.sqrt(2)
                overlap_area = 2 * norm.cdf(-z / 2)
                area_diff = overlap_area * overlap_area
                diff_areas.append(area_diff)

            # 计算传感器分布差的均值
            mean_diff_area = np.mean(diff_areas)

            # 健康指标为1-面积的差值的均值
            health_index = mean_diff_area
            engine_health_index.append(health_index)

            # 更新滑窗索引
            window_indices[condition] += 1

    health_indices.append(engine_health_index)

# 将第一个引擎的健康指标画图
if health_indices:
    plt.plot(health_indices[0])
    plt.xlabel('Cycle')
    plt.ylabel('Health Index')
    plt.title('Health Index of Engine 1')
    plt.show()
else:
    print("No health indices calculated. Please check the data and conditions.")