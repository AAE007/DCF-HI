import pandas as pd

# 读取训练集数据
train_data = pd.read_csv(r'../data/train_FD004.txt', sep=' ', header=None)

# 删除空白列
train_data.dropna(axis=1, how='all', inplace=True)

# 为列命名
train_data.columns = ['unit_number', 'time_in_cycles', 'operational_setting_1',
                      'operational_setting_2', 'operational_setting_3'] + \
                     ['sensor_measurement_' + str(i) for i in range(1, 22)]

# 将训练集数据保存为Excel文件
train_data.to_excel('../data/train_data_FD004.xlsx', sheet_name='Train Data', index=False)

print("训练集数据已成功保存为Excel文件")
