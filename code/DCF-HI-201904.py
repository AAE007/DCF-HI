import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_mat_data(path, key):
    return scipy.io.loadmat(path)[key].astype(float).flatten()


def reshape_lifetime_data(data, columns=5):
    num_rows = len(data) // columns
    reshaped_data = np.zeros((num_rows, columns), dtype=data.dtype)
    for col in range(columns):
        reshaped_data[:, col] = data[col::columns][:num_rows]
    return reshaped_data


def calculate_metrics(predicted, true):
    mae = mean_absolute_error(true, predicted)
    smae = np.mean(np.abs(predicted - true) / (np.abs(true) + np.abs(predicted) + 1e-8))
    rmse = np.sqrt(mean_squared_error(true, predicted))
    diffs = np.diff(predicted)
    monotonicity = np.abs(np.sum(diffs) - np.sum(-diffs)) / (len(diffs) - 1)
    trend = np.corrcoef(predicted, np.linspace(1, 0, len(predicted)))[0, 1]
    robustness = np.mean(np.exp(-np.abs(diffs / (predicted[:-1] + 1e-8))))
    return mae, smae, rmse, monotonicity, trend, robustness


tool_numbers = [5, 7, 13, 4, 9, 15]

health_indices = {}
shock_values = {}
shock_times = {}
distributions = {tool_number: [] for tool_number in tool_numbers}

# Dataframe to store absolute differences and metrics
diff_df = pd.DataFrame()
metrics_list = []

for tool_number in tool_numbers:
    life_data_path = f'../data/Cutting_Tool_{tool_number}_similarity_values.mat'
    lifetime_data = load_mat_data(life_data_path, 'new_feature_map')
    sensor_data = reshape_lifetime_data(lifetime_data, columns=7)

    num_cycles = len(sensor_data)
    true_health_index = np.linspace(1, 0, num_cycles)
    window_size = 30
    engine_health_index = [1] * window_size

    window_1 = [np.array(sensor_data[:window_size])[:, i] for i in range(len(sensor_data[0]))]
    mu_list, std_list = [], []
    for sensor_index in range(len(window_1)):
        mu, std = norm.fit(window_1[sensor_index])
        mu_list.append(mu)
        std_list.append(std)

    mu1 = np.mean(mu_list)
    std1 = np.mean(std_list)
    distributions[tool_number].append((mu1, std1, 'Initial'))

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

        if i % 30 == 0:
            distributions[tool_number].append((mu2, std2, f'Cycle {i}'))

    health_indices[tool_number] = engine_health_index
    shock_values[tool_number] = engine_shock_values
    shock_times[tool_number] = engine_shock_times

    predicted_health_index = np.array(engine_health_index[window_size:])
    true_health_index_segment = true_health_index[window_size:]
    health_index_diff = np.abs(predicted_health_index - true_health_index_segment)

    # Save differences to DataFrame
    diff_df[f'Tool_{tool_number}_Diff'] = health_index_diff

    # Calculate metrics
    mae, smae, rmse, monotonicity, trend, robustness = calculate_metrics(predicted_health_index, true_health_index_segment)
    metrics_list.append({
        'Tool Number': tool_number,
        'MAE': mae,
        'SMAE': smae,
        'RMSE': rmse,
        'Monotonicity': monotonicity,
        'Trend': trend,
        'Robustness': robustness
    })

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    axs[0].plot(range(len(true_health_index)), true_health_index, label='True Health Index')
    axs[0].plot(range(window_size, len(true_health_index)), predicted_health_index, label='Predicted Health Index')
    axs[0].set_xlabel('Cycle')
    axs[0].set_ylabel('Health Index')
    axs[0].set_title(f'Health Index Comparison for Tool {tool_number}')
    axs[0].legend()

    plt.tight_layout()
    plt.savefig(f'../paper/Health_Index_Comparison_Tool_{tool_number}.svg')
    plt.close(fig)

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
    axs[1].bar(range(window_size, len(true_health_index)), health_index_diff, color="#CD1818", width=1.0,
               edgecolor='none')
    axs[1].set_xlabel('Cycle')
    axs[1].set_ylabel('Absolute Difference')
    axs[1].set_title('Absolute Difference between Predicted and True Health Index')

    plt.tight_layout()
    plt.savefig(f'../paper/{tool_number}_health_index_comparison.png', format='png', dpi=600, transparent=True)
    plt.show()
    # 计算动态上限
    cumulative_shock_upper = []
    decay_factor = 0.97

    for i in range(len(predicted_health_index)):
        if i == 0:
            cumulative_shock_upper.append(predicted_health_index[i])
        else:
            decay = decay_factor ** (i + 1)
            cumulative_shock_upper.append(min(cumulative_shock_upper[-1], cumulative_shock_upper[-1] + (predicted_health_index[i] - cumulative_shock_upper[-1]) * decay))

    # 绘制变分正则化值和冲击标记
    if var_reg_values:
        plt.figure(figsize=(12, 6))
        plt.plot(var_reg_values, label='Variation Regularization')
        plt.plot(upper_limits, 'r--', label='Upper 2σ Limit')
        plt.plot(lower_limits, 'g--', label='Lower 2σ Limit')
        if shock_times[tool_number]:
            plt.scatter(shock_times[tool_number], [var_reg_values[i] for i in shock_times[tool_number]], color='r', label='Shock')
        plt.xlabel('Cycle')
        plt.ylabel('Variation Regularization')
        plt.title(f'Variation Regularization with Shock Marks and 2σ Limits for Tool {tool_number}')
        plt.legend()
        plt.savefig(f'../paper/Tool_{tool_number}_Shock_Marks.svg', format='svg')
        plt.close()

        # 绘制冲击值的累积图（阶梯图）
        if shock_values[tool_number]:
            cumulative_shocks = np.cumsum(shock_values[tool_number])
            steps = np.concatenate([[0], cumulative_shocks])
            shock_steps = np.concatenate([[0], shock_times[tool_number]])

            # 填充最后的值以确保长度一致
            last_shock_time = shock_steps[-1]
            steps_filled = np.concatenate([steps, [steps[-1]] * (num_cycles - last_shock_time - 1)])
            shock_steps_filled = np.concatenate([shock_steps, range(last_shock_time + 1, num_cycles)])

            plt.figure(figsize=(12, 6))
            plt.step(shock_steps_filled, steps_filled, where='post', label='Cumulative Shock Values')
            plt.plot(range(window_size, num_cycles), cumulative_shock_upper, 'b--', label='Upper Limit')
            plt.scatter(shock_steps[1:], steps[1:], color='r', label='Shock')
            plt.xlabel('Cycle')
            plt.ylabel('Cumulative Shock Values')
            plt.title(f'Cumulative Shock Values for Tool {tool_number}')
            plt.legend()
            plt.savefig(f'../paper/Tool_{tool_number}_Shock_Values.svg', format='svg')
            plt.close()

# Save absolute differences to Excel
diff_df.to_excel('../paper/201904_Health_Index_Differences.xlsx', index=False)

# Save metrics to Excel
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_excel('../paper/201904_Health_Index_Metrics.xlsx', index=False)
