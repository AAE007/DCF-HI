import scipy.io
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

# 转换时间格式，将字符串转换成 datetime 格式
def convert_to_time(hmm):
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

# 加载 mat 文件
def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    filename = matfile.split('/')[-1].split('.')[0]
    col = data[filename]
    col = col[0][0][0][0]
    size = col.shape[0]

    data = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        if str(col[i][0][0]) != 'impedance':
            for j in range(len(k)):
                t = col[i][3][0][0][j][0]
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        d1['type'], d1['temp'], d1['time'], d1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(convert_to_time(col[i][2][0])), d2
        data.append(d1)

    return data

# 提取锂电池容量
def getBatteryCapacity(Battery):
    cycle, capacity = [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'discharge':
            capacity.append(Bat['data']['Capacity'][0])
            cycle.append(i)
            i += 1
    return [cycle, capacity]

# 获取锂电池充电或放电时的测试数据
def getBatteryValues(Battery, Type='charge'):
    data = []
    for Bat in Battery:
        if Bat['type'] == Type:
            data.append(Bat['data'])
    return data


Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']  # 4 个数据集的名字
dir_path = r'../data/1. BatteryAgingARC-FY08Q4/'

capacity, charge, discharge = {}, {}, {}
for name in Battery_list:
    print('Load Dataset ' + name + '.mat ...')
    path = dir_path + name + '.mat'
    data = loadMat(path)
    capacity[name] = getBatteryCapacity(data)  # 放电时的容量数据 真实容量
    charge[name] = getBatteryValues(data, 'charge')  # 充电数据 测试数据
    discharge[name] = getBatteryValues(data, 'discharge')  # 放电数据 测试数据


# 定义正则化函数
def normalize_0_1(arr, min_val, max_val):
    return (arr - min_val) / (max_val - min_val)


# 初始化一个新的字典用于存储拼接后的数据
combined_data = {}
# 列出所有电池名称
Battery_list = charge.keys()
# 对每个电池的数据进行处理
for name in Battery_list:
    # 选择最短的长度
    min_length = min(len(charge[name]), len(discharge[name]))

    # 初始化一个新的列表用于存储每个电池拼接后的数据
    combined_list = []
    # charge_key = ['Voltage_measured', 'Current_measured', 'Current_charge']
    charge_key = []
    discharge_key = ['Voltage_measured', 'Current_measured', 'Current_load', 'Voltage_load', 'Time', 'Capacity']

    # 遍历最短长度，拼接字典中的键值
    for i in range(min_length):
        combined_dict = {}
        for key in charge_key:
            # 将列表转换为ndarray
            arrays = np.array(charge[name][i][key])
            # 检查并删除NaN和空值
            arrays = arrays[np.isfinite(arrays)]
            # 计算RMS
            rms_values = np.sqrt(np.mean(arrays ** 2, axis=0))
            # 将RMS值存储回combined_dict
            combined_dict[f'charge_{key}'] = rms_values

        for key in discharge_key:
            # 将列表转换为ndarray
            arrays = np.array(discharge[name][i][key])
            # 检查并删除NaN和空值
            arrays = arrays[np.isfinite(arrays)]
            # 计算RMS
            rms_values = np.sqrt(np.mean(arrays ** 2, axis=0))
            # 将RMS值存储回combined_dict
            combined_dict[f'discharge_{key}'] = rms_values

        combined_list.append(combined_dict)

    # 将拼接后的数据放入顶层字典中
    combined_data[name] = combined_list

# 进行正则化
scaler = MinMaxScaler()

for name in combined_data:
    combined_list = combined_data[name]

    # 将所有键值的数据拼接到一个数组中以便正则化
    keys = combined_list[0].keys()
    all_data = {key: [] for key in keys}

    for entry in combined_list:
        for key in keys:
            all_data[key].append(entry[key])

    for key in keys:
        # 转换为numpy数组并进行0-1正则化
        data = np.array(all_data[key]).reshape(-1, 1)
        normalized_data = scaler.fit_transform(data).flatten()

        # 将正则化后的数据放回到combined_list中
        for i in range(len(combined_list)):
            combined_list[i][key] = normalized_data[i]

    combined_data[name] = combined_list

# 将engine_data转换为list嵌套list
engine_data_list = {}
for engine_id, engine_data in combined_data.items():
    engine_list = []
    for data in engine_data:
        engine_list.append([data[key] for key in data])
    engine_data_list[engine_id] = engine_list

# 初始化健康指标和冲击值的存储
health_indices = {}
shock_values = {}
shock_times = {}

# 对每个引擎逐个处理
for engine_id, engine_data in engine_data_list.items():
    num_cycles = len(engine_data)
    true_health_index = np.linspace(1, 0, num_cycles)
    sensor_data = engine_data
    # 定义初始窗口
    window_size = 20
    engine_health_index = [1] * window_size  # 初始化前20个健康指标为1

    # 创建一个包含len(win_data)个空ndarray的列表
    window_1 = [np.empty((0,)) for _ in range(len(sensor_data[1]))]

    # 拼接数据
    for win_data in sensor_data[:window_size]:
        for i in range(len(win_data)):
            window_1[i] = np.hstack((window_1[i], win_data[i]))

    for sensor_index in range(len(window_1)):
        data_1 = window_1[sensor_index]
        mu1, std1 = norm.fit(data_1)

    var_reg_values = []
    upper_limits = []
    lower_limits = []
    engine_shock_values = []
    engine_shock_times = []

    for i in range(window_size, num_cycles):
        diff_areas = []
        window_2 = np.array(sensor_data[i])
        # 对每个传感器数据进行贝叶斯高斯分布拟合并计算面积差
        mu2, std2 = norm.fit(window_2)
        # 计算两个分布的面积差值
        area_diff = np.abs(norm.cdf(mu1, mu2, std2) - norm.cdf(mu2, mu1, std1))
        diff_areas.append(area_diff)

        # 计算传感器分布差的均值
        mean_diff_area = np.mean(diff_areas)

        # 当前窗口的健康指标
        current_health_index = 1 - mean_diff_area

        # 平滑处理
        if len(engine_health_index) >= 5:
            smoothed_health_index = 0.5 * current_health_index + 0.5 * np.mean(engine_health_index[-5:])
        else:
            smoothed_health_index = current_health_index

        engine_health_index.append(smoothed_health_index)


        # 计算变分正则化值（标准化处理）
        diff_flat = np.diff(window_2)
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

    health_indices[engine_id] = engine_health_index
    shock_values[engine_id] = engine_shock_values
    shock_times[engine_id] = engine_shock_times

    # 生成的健康指标与真实健康指标的差值的绝对值
    predicted_health_index = np.array(engine_health_index[window_size:])  # 从第31步开始
    true_health_index_segment = np.linspace(1, 0, num_cycles)[window_size:]
    health_index_diff = np.abs(predicted_health_index - true_health_index_segment)

    # 绘制结果图
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    # 第一个子图：健康指标曲线
    axs[0].plot(range(len(true_health_index)), true_health_index, label='True Health Index')
    axs[0].plot(range(window_size, len(true_health_index)), predicted_health_index, label='Predicted Health Index')
    axs[0].set_xlabel('Cycle')
    axs[0].set_ylabel('Health Index')
    axs[0].set_title(f'Health Index Comparison for {engine_id}')
    axs[0].legend()

    # 第二个子图：健康指标差值的绝对值柱状图
    axs[1].bar(range(window_size, len(true_health_index)), health_index_diff)
    axs[1].set_xlabel('Cycle')
    axs[1].set_ylabel('Absolute Difference')
    axs[1].set_title(f'Absolute Difference between Predicted and True Health Index for {engine_id}')

    plt.tight_layout()
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
        if shock_times[engine_id]:
            plt.scatter(shock_times[engine_id], [var_reg_values[i] for i in shock_times[engine_id]], color='r', label='Shock')
        plt.xlabel('Cycle')
        plt.ylabel('Variation Regularization')
        plt.title(f'Variation Regularization with Shock Marks and 2σ Limits for {engine_id}')
        plt.legend()
        plt.show()

        # 绘制冲击值的累积图（阶梯图）
        if shock_values[engine_id]:
            cumulative_shocks = np.cumsum(shock_values[engine_id])
            steps = np.concatenate([[0], cumulative_shocks])
            shock_steps = np.concatenate([[0], shock_times[engine_id]])

            # 延长最后一个冲击时间步以确保长度一致
            last_shock_time = shock_steps[-1] if len(shock_steps) > 1 else len(engine_data)
            steps_filled = np.concatenate([steps, [steps[-1]] * (len(engine_data) - last_shock_time - 1)])
            shock_steps_filled = np.concatenate([shock_steps, range(last_shock_time + 1, len(engine_data))])

            plt.figure(figsize=(12, 6))
            plt.step(shock_steps_filled, steps_filled, where='post', label='Cumulative Shock Values')
            plt.plot(range(window_size, len(engine_data)), cumulative_shock_upper, label='Upper Limit', linestyle='--')
            plt.scatter(shock_steps[1:], steps[1:], color='r', label='Shock')
            plt.xlabel('Cycle')
            plt.ylabel('Cumulative Shock Values')
            plt.title(f'Cumulative Shock Values for {engine_id}')
            plt.legend()
            plt.show()
    else:
        print(f"No variation regularization values calculated for {engine_id}.")

# 初始化一个包含所有键的空ndarray的字典
keys = engine_data[0].keys()
merged_data = {key: [] for key in keys}

# 收集数据
for data in engine_data:
    for key in keys:
        merged_data[key].append(data[key])

# 将列表转换为ndarray并进行拼接
for key in keys:
    merged_data[key] = np.concatenate(merged_data[key], axis=0)

# 计算每个ndarray的RMS并进行拼接
for key in keys:
    # 将列表转换为ndarray
    arrays = np.array(merged_data[key])

    # 计算RMS
    rms_values = np.sqrt(np.mean(arrays ** 2, axis=0))

    # 将RMS值存储回merged_data
    merged_data[key] = rms_values

# 初始化一个包含所有键的空列表的字典
keys = engine_data[0].keys()
merged_data = {key: [] for key in keys}

# 收集数据
for data in engine_data:
    for key in keys:
        merged_data[key].append(data[key])

# 计算RMS值并拼接
rms_data = {key: np.empty((0,)) for key in keys}

for key in keys:
    rms_values = [np.sqrt(np.mean(np.square(arr))) for arr in merged_data[key]]
    rms_data[key] = np.array(rms_values)

# 画折线图
for key, values in rms_data.items():
    plt.figure()
    plt.plot(values, marker='o')
    plt.title(f'RMS Values for {key}')
    plt.xlabel('Sample Index')
    plt.ylabel('RMS Value')
    plt.grid(True)
    plt.show()
