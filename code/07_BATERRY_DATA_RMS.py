import scipy.io
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
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
    charge_key = ['Voltage_measured', 'Current_measured', 'Current_charge']
    discharge_key = ['Voltage_measured', 'Current_measured', 'Current_load', 'Voltage_load', 'Time', 'Capacity']

    # 遍历最短长度，拼接字典中的键值
    for i in range(min_length):
        combined_dict = {}
        # for key in charge_key:
        for key in charge[name][i].keys():
            # 将列表转换为ndarray
            arrays = np.array(charge[name][i][key])
            # 检查并删除NaN和空值
            arrays = arrays[np.isfinite(arrays)]
            # 计算RMS
            rms_values = np.sqrt(np.mean(arrays ** 2, axis=0))
            # 将RMS值存储回combined_dict
            combined_dict[f'charge_{key}'] = rms_values

        # for key in discharge_key:
        for key in discharge[name][i].keys():

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

engine_data = combined_data["B0018"]

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