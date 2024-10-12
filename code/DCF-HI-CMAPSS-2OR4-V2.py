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
shock_values = []
shock_times = []
hi_diff_means = []  # 新增列表存储差值绝对值均值

# 获取所有的引擎编号
engine_ids = train_data['unit_number'].unique()[19:20]

# 定义操作条件阈值及其范围
op_conditions = [0, 10, 20, 25, 35, 42]
condition_ranges = [(condition - 3, condition + 3) for condition in op_conditions]
window_size = 30
# 对每个引擎逐个处理
for engine_id in engine_ids:
    # 过滤出当前引擎的数据
    engine_data = train_data[train_data['unit_number'] == engine_id]
    num_cycles = len(engine_data)

    # 生成真实健康指标
    true_health_index = np.linspace(1, 0, num_cycles)

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

    engine_health_index = [1] * window_size  # 初始化前30个健康指标为1
    engine_shock_values = []
    engine_shock_times = []

    # 初始化滑窗索引
    window_indices = {condition: 0 for condition in grouped_data.keys()}

    var_reg_values = []
    upper_limits = []
    lower_limits = []

    # 定义初始窗口
    initial_window = {condition: grouped_data[condition][:5] for condition in window_indices.keys()}

    # 循环滑窗，直到所有条件数据滑动完
    while window_indices:
        for condition in list(window_indices.keys()):
            if window_indices[condition] + 6 > len(grouped_data[condition]):
                del window_indices[condition]
                continue
            window_1 = initial_window[condition]
            window_2 = grouped_data[condition][window_indices[condition]:window_indices[condition] + 6]

            # 对每个传感器数据进行贝叶斯高斯分布拟合
            diff_areas = []
            for sensor_index in range(window_1.shape[1]):
                if np.std(window_1[:, sensor_index]) == 0 or np.std(window_2[:, sensor_index]) == 0:
                    area_diff = 0
                else:
                    mu1, std1 = norm.fit(window_1[:, sensor_index])
                    mu2, std2 = norm.fit(window_2[:, sensor_index])

                    # 计算两个分布的重叠面积
                    diff_mu = np.abs(mu1 - mu2)
                    combined_sigma = np.sqrt(std1 ** 2 + std2 ** 2)
                    z = diff_mu / combined_sigma * np.sqrt(2)
                    overlap_area = 2 * norm.cdf(-z / 2)
                    area_diff = overlap_area * overlap_area

                diff_areas.append(area_diff)

            # 计算传感器分布差的均值
            mean_diff_area = np.mean(diff_areas)

            # 当前窗口的健康指标
            current_health_index = mean_diff_area

            # 平滑处理
            if len(engine_health_index) >= 5:
                smoothed_health_index = 0.5 * current_health_index + 0.5 * np.mean(engine_health_index[-5:])
            else:
                smoothed_health_index = current_health_index

            engine_health_index.append(smoothed_health_index)

            # 窗口数据处理为1行
            window_2_flat = window_2.flatten()
            # 计算变分正则化值（标准化处理）
            diff_flat = np.diff(window_2_flat)
            var_reg_value = np.sum(np.abs(diff_flat)) / len(diff_flat)
            var_reg_values.append(var_reg_value)

            # 计算前30个窗口的2sigma的上下限
            if len(var_reg_values) > window_size:
                recent_values = var_reg_values[-window_size:]
                mean_recent = np.mean(recent_values)
                std_recent = np.std(recent_values)
                upper_limit = mean_recent + 2 * std_recent
                lower_limit = mean_recent - 2 * std_recent
                upper_limits.append(upper_limit)
                lower_limits.append(lower_limit)

                # 检查当前步的变分正则化值是否超出上下限
                if var_reg_value > upper_limit or var_reg_value < lower_limit:
                    shock_value = var_reg_value - upper_limit if var_reg_value > upper_limit else lower_limit - var_reg_value
                    if not engine_shock_values or len(var_reg_values) - 1 - engine_shock_times[-1] > 5:
                        engine_shock_values.append(shock_value)
                        engine_shock_times.append(len(var_reg_values) - 1)
                    else:
                        if shock_value > engine_shock_values[-1]:
                            engine_shock_values[-1] = shock_value
                            engine_shock_times[-1] = len(var_reg_values) - 1
            else:
                upper_limits.append(np.nan)
                lower_limits.append(np.nan)

            # 更新滑窗索引
            window_indices[condition] += 1

    health_indices.append(engine_health_index)
    shock_values.append(engine_shock_values)
    shock_times.append(engine_shock_times)

    # 生成的健康指标与真实健康指标的差值的绝对值
    predicted_health_index = np.array(engine_health_index[window_size:])  # 从第31步开始
    true_health_index_segment = true_health_index[window_size:]
    health_index_diff = np.abs(predicted_health_index - true_health_index_segment)
    hi_diff_mean = np.mean(health_index_diff)  # 计算差值绝对值的均值
    hi_diff_means.append(hi_diff_mean)  # 将结果添加到列表中

# 保存结果到新的 Excel 文件
output_df = pd.DataFrame({
    'Engine_ID': engine_ids,
    'HI_Diff_Mean': hi_diff_means
})
output_df.to_excel('../paper/FD004_hi_diff_means.xlsx', index=False)

# 绘制结果图
fig, axs = plt.subplots(2, 1, figsize=(12, 3))

# 第一个子图：健康指标曲线
axs[0].plot(range(len(true_health_index)), true_health_index, label='True Health Index')
axs[0].plot(range(window_size, len(true_health_index)), predicted_health_index, label='Predicted Health Index')
axs[0].set_xlabel('Cycle')
axs[0].set_ylabel('Health Index')
axs[0].set_title('Health Index Comparison')
axs[0].legend()

# 第二个子图：健康指标差值的绝对值柱状图
axs[1].bar(range(window_size, len(true_health_index)), health_index_diff, color="#CD1818", width=1.0, edgecolor='none')
axs[1].set_xlabel('Cycle')
axs[1].set_ylabel('Absolute Difference')
axs[1].set_title('Absolute Difference between Predicted and True Health Index')

plt.tight_layout()
plt.savefig(f'../paper/{engine_id}_health_index_comparison.png', format='png', dpi=600, transparent=True)
plt.savefig(f'../paper/{engine_id}_health_index_comparison.svg', format='svg')
plt.show()

# 累计冲击图的上限
cumulative_shock_upper = []
decay_factor = 0.97

for i in range(len(predicted_health_index)):
    if i == 0:
        cumulative_shock_upper.append(predicted_health_index[i])
    else:
        decay = decay_factor ** (i + 1)
        cumulative_shock_upper.append(min(cumulative_shock_upper[-1], cumulative_shock_upper[-1] + (predicted_health_index[i]-cumulative_shock_upper[-1]) * decay))

# 绘制变分正则化值和冲击标记
if var_reg_values:
    plt.figure(figsize=(12, 6))
    plt.plot(var_reg_values, label='Variation Regularization')
    plt.plot(upper_limits, 'r--', label='Upper 2σ Limit')
    plt.plot(lower_limits, 'g--', label='Lower 2σ Limit')
    if shock_times:
        plt.scatter(shock_times[0], [var_reg_values[i] for i in shock_times[0]], color='r', label='Shock')
    plt.xlabel('Cycle')
    plt.ylabel('Variation Regularization')
    plt.title('Variation Regularization with Shock Marks and 2σ Limits')
    plt.legend()
    plt.savefig(f'../paper/{engine_id}_Shock_Marks.svg', format='svg')
    plt.show()

    # 绘制冲击值的累积图（阶梯图）
    if shock_values[0]:
        cumulative_shocks = np.cumsum(shock_values[0])
        steps = np.concatenate([[0], cumulative_shocks])
        shock_steps = np.concatenate([[0], shock_times[0]])

        # 填充最后的值以确保长度一致
        last_shock_time = shock_steps[-1]
        steps_filled = np.concatenate([steps, [steps[-1]] * (len(engine_data) - last_shock_time - 1)])
        shock_steps_filled = np.concatenate([shock_steps, range(last_shock_time + 1, len(engine_data))])

        plt.figure(figsize=(12, 6))
        plt.step(shock_steps_filled, steps_filled, where='post', label='Cumulative Shock Values')
        plt.plot(range(window_size, len(engine_data)), cumulative_shock_upper, label='Upper Limit', linestyle='--')
        plt.scatter(shock_steps[1:], steps[1:], color='r', label='Shock')
        plt.xlabel('Cycle')
        plt.ylabel('Cumulative Shock Values')
        plt.title('Cumulative Shock Values of Engine 1')
        plt.legend()
        plt.savefig(f'../paper/{engine_id}_Shock_Values.svg', format='svg')
        plt.show()
else:
    print("No variation regularization values calculated. Please check the data and conditions.")