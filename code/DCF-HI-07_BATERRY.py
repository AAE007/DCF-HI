import scipy.io
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler


def convert_to_time(hmm):
    return datetime(int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5]))


def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    filename = matfile.split('/')[-1].split('.')[0]
    col = data[filename][0][0][0][0]
    size = col.shape[0]

    data = []
    for i in range(size):
        if str(col[i][0][0]) != 'impedance':
            fields = list(col[i][3][0].dtype.fields.keys())
            d2 = {k: [col[i][3][0][0][j][0][m] for m in range(len(col[i][3][0][0][j][0]))] for j, k in
                  enumerate(fields)}
            d1 = {'type': str(col[i][0][0]), 'temp': int(col[i][1][0]), 'time': str(convert_to_time(col[i][2][0])),
                  'data': d2}
            data.append(d1)
    return data


def getBatteryCapacity(Battery):
    return [[i + 1 for i in range(len(Battery)) if Battery[i]['type'] == 'discharge'],
            [Battery[i]['data']['Capacity'][0] for i in range(len(Battery)) if Battery[i]['type'] == 'discharge']]


def getBatteryValues(Battery, Type='charge'):
    return [Bat['data'] for Bat in Battery if Bat['type'] == Type]


Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
dir_path = r'../data/1. BatteryAgingARC-FY08Q4/'

capacity, charge, discharge = {}, {}, {}
for name in Battery_list:
    print(f'Load Dataset {name}.mat ...')
    data = loadMat(f'{dir_path}{name}.mat')
    capacity[name] = getBatteryCapacity(data)
    charge[name] = getBatteryValues(data, 'charge')
    discharge[name] = getBatteryValues(data, 'discharge')


def calculate_rms(arrays):
    arrays = arrays[np.isfinite(arrays)]
    return np.sqrt(np.mean(arrays ** 2))


combined_data = {}
for name in Battery_list:
    min_length = min(len(charge[name]), len(discharge[name]))
    charge_key = ['Voltage_measured', 'Current_measured', 'Current_charge']
    # charge_key = []
    discharge_key = ['Voltage_measured', 'Current_measured', 'Current_load', 'Voltage_load', 'Time', 'Capacity']

    combined_list = []
    for i in range(min_length):
        combined_dict = {}
        # for key in charge_key:
        for key in charge[name][i].keys():
            combined_dict[f'charge_{key}'] = calculate_rms(np.array(charge[name][i][key]))
        # for key in discharge_key:
        for key in discharge[name][i].keys():
            combined_dict[f'discharge_{key}'] = calculate_rms(np.array(discharge[name][i][key]))
        combined_list.append(combined_dict)
    combined_data[name] = combined_list

scaler = MinMaxScaler()
for name in combined_data:
    combined_list = combined_data[name]
    keys = combined_list[0].keys()
    all_data = {key: [entry[key] for entry in combined_list] for key in keys}
    for key in keys:
        data = np.array(all_data[key]).reshape(-1, 1)
        normalized_data = scaler.fit_transform(data).flatten()
        for i in range(len(combined_list)):
            combined_list[i][key] = normalized_data[i]
    combined_data[name] = combined_list

engine_data_list = {engine_id: [[data[key] for key in data] for data in engine_data]
                    for engine_id, engine_data in combined_data.items()}

health_indices = {}
shock_values = {}
shock_times = {}
distributions = {engine_id: [] for engine_id in engine_data_list}

for engine_id, sensor_data in engine_data_list.items():
    num_cycles = len(sensor_data)
    true_health_index = np.linspace(1, 0, num_cycles)
    window_size = 30
    engine_health_index = [1] * window_size

    # 初始化window_1
    window_1 = [np.array(sensor_data[:window_size])[:, i] for i in range(len(sensor_data[0]))]

    # 计算前window_size个窗口的贝叶斯正则化均值
    mu_list, std_list = [], []
    for sensor_index in range(len(window_1)):
        mu, std = norm.fit(window_1[sensor_index])
        mu_list.append(mu)
        std_list.append(std)

    mu1 = np.mean(mu_list)
    std1 = np.mean(std_list)
    distributions[engine_id].append((mu1, std1, 'Initial'))

    var_reg_values, upper_limits, lower_limits = [], [], []
    engine_shock_values, engine_shock_times = [], []

    for i in range(window_size, num_cycles):
        window_2 = np.array(sensor_data[i])
        mu2, std2 = norm.fit(window_2)
        # 计算两个分布的重叠面积
        diff_mu = np.abs(mu1 - mu2)
        combined_sigma = np.sqrt(std1 ** 2 + std2 ** 2)
        z = diff_mu / combined_sigma * np.sqrt(2)
        mean_diff_area = 2 * norm.cdf(-z / 2)
        current_health_index = mean_diff_area * mean_diff_area
        smoothed_health_index = (0.5 * current_health_index + 0.5 * np.mean(engine_health_index[-5:])) if len(
            engine_health_index) >= 5 else current_health_index
        engine_health_index.append(smoothed_health_index)

        diff_flat = np.diff(window_2)
        var_reg_value = np.sum(np.abs(diff_flat)) / len(diff_flat)
        var_reg_values.append(var_reg_value)

        if len(var_reg_values) > window_size:
            recent_values = var_reg_values[-window_size:]
            mean_recent, std_recent = np.mean(recent_values), np.std(recent_values)
            upper_limit, lower_limit = mean_recent + 2 * std_recent, mean_recent - 2 * std_recent
            upper_limits.append(upper_limit)
            lower_limits.append(lower_limit)

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

        if i % 30 == 0:  # 每隔10步记录一次分布
            distributions[engine_id].append((mu2, std2, f'Cycle {i}'))

    health_indices[engine_id] = engine_health_index
    shock_values[engine_id] = engine_shock_values
    shock_times[engine_id] = engine_shock_times

    predicted_health_index = np.array(engine_health_index[window_size:])
    true_health_index_segment = true_health_index[window_size:]
    health_index_diff = np.abs(predicted_health_index - true_health_index_segment)

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    axs[0].plot(range(len(true_health_index)), true_health_index, label='True Health Index')
    axs[0].plot(range(window_size, len(true_health_index)), predicted_health_index, label='Predicted Health Index')
    axs[0].set_xlabel('Cycle')
    axs[0].set_ylabel('Health Index')
    axs[0].set_title(f'Health Index Comparison for {engine_id}')
    axs[0].legend()

    axs[1].bar(range(window_size, len(true_health_index)), health_index_diff)
    axs[1].set_xlabel('Cycle')
    axs[1].set_ylabel('Absolute Difference')
    axs[1].set_title(f'Absolute Difference between Predicted and True Health Index for {engine_id}')

    plt.tight_layout()
    plt.show()

    cumulative_shock_upper = [predicted_health_index[0]]
    decay_factor = 0.97
    for i in range(1, len(predicted_health_index)):
        decay = decay_factor ** (i + 1)
        cumulative_shock_upper.append(min(cumulative_shock_upper[-1], cumulative_shock_upper[-1] + (
                    predicted_health_index[i] - cumulative_shock_upper[-1]) * decay))

    if var_reg_values:
        plt.figure(figsize=(12, 6))
        plt.plot(var_reg_values, label='Variation Regularization')
        plt.plot(upper_limits, 'r--', label='Upper 2σ Limit')
        plt.plot(lower_limits, 'g--', label='Lower 2σ Limit')
        if shock_times[engine_id]:
            plt.scatter(shock_times[engine_id], [var_reg_values[j] for j in shock_times[engine_id]], color='r',
                        label='Shock')
        plt.xlabel('Cycle')
        plt.ylabel('Variation Regularization')
        plt.title(f'Variation Regularization with Shock Marks and 2σ Limits for {engine_id}')
        plt.legend()
        plt.show()

        if shock_values[engine_id]:
            cumulative_shocks = np.cumsum(shock_values[engine_id])
            steps = np.concatenate([[0], cumulative_shocks])
            shock_steps = np.concatenate([[0], shock_times[engine_id]])
            last_shock_time = shock_steps[-1] if len(shock_steps) > 1 else len(sensor_data)
            steps_filled = np.concatenate([steps, [steps[-1]] * (len(sensor_data) - last_shock_time - 1)])
            shock_steps_filled = np.concatenate([shock_steps, range(last_shock_time + 1, len(sensor_data))])

            plt.figure(figsize=(12, 6))
            plt.step(shock_steps_filled, steps_filled, where='post', label='Cumulative Shock Values')
            plt.plot(range(window_size, len(sensor_data)), cumulative_shock_upper, label='Upper Limit', linestyle='--')
            plt.scatter(shock_steps[1:], steps[1:], color='r', label='Shock')
            plt.xlabel('Cycle')
            plt.ylabel('Cumulative Shock Values')
            plt.title(f'Cumulative Shock Values for {engine_id}')
            plt.legend()
            plt.show()
    else:
        print(f"No variation regularization values calculated for {engine_id}.")

# 绘制每个电池的贝叶斯正态分布图并打印面积
x = np.linspace(-3, 3, 1000)
for engine_id, dist_list in distributions.items():
    plt.figure(figsize=(12, 6))
    for mu, std, label in dist_list:
        y = norm.pdf(x, mu, std)
        area = np.trapz(y, x)
        print(f'{engine_id} - {label}: Area under the curve = {area}')
        plt.plot(x, y, label=label)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.title(f'Bayesian Gaussian Distributions for {engine_id}')
    plt.legend()
    plt.show()
