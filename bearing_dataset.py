import torch
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from joblib import dump
from sklearn.model_selection import train_test_split

# 使用相对路径直接指向 matfiles 文件夹
base_directory = 'dataset/matfiles'  # 相对路径

file_names = [
    '0_0.mat', '7_1.mat', '7_2.mat', '7_3.mat', 
    '14_1.mat', '14_2.mat', '14_3.mat', 
    '21_1.mat', '21_2.mat', '21_3.mat'
]

data_columns = ['X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time', 
                'X169_DE_time', 'X185_DE_time', 'X197_DE_time', 'X209_DE_time', 
                'X222_DE_time', 'X234_DE_time']

columns_name = ['de_normal', 'de_7_inner', 'de_7_ball', 'de_7_outer', 
                'de_14_inner', 'de_14_ball', 'de_14_outer', 
                'de_21_inner', 'de_21_ball', 'de_21_outer']

# 生成 data_12k_10c.csv 的数据
data_12k_10c = pd.DataFrame()
for index in range(10):
    # 读取MAT文件
    data = loadmat(f'{base_directory}/{file_names[index]}') # 使用斜杠作为路径分隔符，兼容不同操作系统
    dataList = data[data_columns[index]].reshape(-1)
    data_12k_10c[columns_name[index]] = dataList[:119808]  # 假设每列数据有 119808 个点

# 将数据保存为 CSV 文件
data_12k_10c.to_csv('data_12k_10c.csv', index=False)
print("✅ data_12k_10c.csv 已生成，形状:", data_12k_10c.shape)

# 标签映射：将列名映射为故障类型（0: 正常, 1: 滚动体故障, 2: 内圈故障, 3: 外圈故障）
label_map = {
    'de_normal': 0,
    'de_7_inner': 2, 'de_14_inner': 2, 'de_21_inner': 2,
    'de_7_ball': 1, 'de_14_ball': 1, 'de_21_ball': 1,
    'de_7_outer': 3, 'de_14_outer': 3, 'de_21_outer': 3
}

# 带通滤波器设计
def bandpass_filter(data, lowcut=200, highcut=5900, fs=12000, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# 数据归一化
def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val + 1e-8)

# 构造样本函数
def create_dualtask_samples(data, label, condition, Lhist=1024, Lpred=1024, stride=512):
    L = Lhist + Lpred
    samples = []
    conditions = []
    for i in range(0, len(data) - L + 1, stride):
        x = data[i: i + Lhist]
        y = data[i + Lhist: i + L]
        samples.append(np.concatenate([x, y, [label]]))
        conditions.append(condition)
    return np.array(samples), np.array(conditions)

# 定义 make_dualtask_data_labels 函数
def make_dualtask_data_labels(dataframe, Lhist=1024, Lpred=1024):
    """
    将 DataFrame 转换为多任务学习所需的输入和输出张量
    
    参数:
        dataframe: 包含样本的 DataFrame
        Lhist: 历史时间步数
        Lpred: 预测时间步数
    
    返回:
        x_tensor: 输入序列 (N, Lhist)
        y_trend_tensor: 预测目标 (N, Lpred)
        y_class_tensor: 分类标签 (N,)
    """
    input_end = Lhist
    trend_end = Lhist + Lpred

    x_data = dataframe.iloc[:, 0:input_end]
    y_trend = dataframe.iloc[:, input_end:trend_end]
    y_class = dataframe.iloc[:, -1]

    x_tensor = torch.tensor(x_data.values).float()
    y_trend_tensor = torch.tensor(y_trend.values).float()
    y_class_tensor = torch.tensor(y_class.values.astype('int64'))

    return x_tensor, y_trend_tensor, y_class_tensor

# 主函数：处理数据集并均匀划分
def process_dataset(filename, Lhist=1024, Lpred=1024, stride=512, split_rate=[0.6, 0.2, 0.2]):
    """
    处理数据集并按比例均匀划分为训练集、验证集和测试集
    
    参数:
        filename: 输入的 CSV 文件路径
        Lhist: 历史时间步数
        Lpred: 预测时间步数
        stride: 滑动窗口步幅
        split_rate: 数据集划分比例 [训练集, 验证集, 测试集]
    
    返回:
        train_df, val_df, test_df: 划分后的数据集 (DataFrame 格式)
    """
    # 读取数据
    origin_data = pd.read_csv(filename)
    all_samples = []
    all_conditions = []

    # 处理每一列数据
    for col_name in origin_data.columns:
        if col_name not in label_map:
            continue
        label = label_map[col_name]
        col_data = origin_data[col_name].values
        filtered = bandpass_filter(col_data)
        normed = normalize(filtered)
        samples, conditions = create_dualtask_samples(normed, label, col_name, Lhist, Lpred, stride)
        all_samples.append(samples)
        all_conditions.append(conditions)

    # 合并所有样本
    total_samples = np.vstack(all_samples)
    total_conditions = np.concatenate(all_conditions)

    # 分层抽样划分数据集
    train_idx, temp_idx = train_test_split(
        np.arange(len(total_samples)),
        test_size=(split_rate[1] + split_rate[2]),  # 验证集+测试集比例
        stratify=total_conditions,
        random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=split_rate[2] / (split_rate[1] + split_rate[2]),  # 测试集占临时集的比例
        stratify=total_conditions[temp_idx],
        random_state=42
    )

    # 根据索引提取子集
    train_set = total_samples[train_idx]
    val_set = total_samples[val_idx]
    test_set = total_samples[test_idx]

    # 转换为 DataFrame
    col_count = Lhist + Lpred
    columns = [f'feat_{i}' for i in range(col_count)] + ['label']
    train_df = pd.DataFrame(train_set, columns=columns)
    val_df = pd.DataFrame(val_set, columns=columns)
    test_df = pd.DataFrame(test_set, columns=columns)

    # 保存为 .joblib 文件
    dump(train_df, 'train_set.joblib')
    dump(val_df, 'val_set.joblib')
    dump(test_df, 'test_set.joblib')
    print("✅ 数据处理与均匀划分完毕，数据集已保存为 .joblib 文件")

    return train_df, val_df, test_df

# 执行数据处理
if __name__ == "__main__":
    # 处理数据集并划分
    train_df, val_df, test_df = process_dataset('data_12k_10c.csv')
    print("训练集样本数:", len(train_df))
    print("验证集样本数:", len(val_df))
    print("测试集样本数:", len(test_df))

    # 使用定义的 make_dualtask_data_labels 函数进行张量转换
    train_X, train_Ytrend, train_Yclass = make_dualtask_data_labels(train_df)
    val_X, val_Ytrend, val_Yclass = make_dualtask_data_labels(val_df)
    test_X, test_Ytrend, test_Yclass = make_dualtask_data_labels(test_df)

    # 保存张量为 .joblib 文件
    dump(train_X, 'trainX_dualtask.joblib')
    dump(train_Ytrend, 'trainYtrend_dualtask.joblib')
    dump(train_Yclass, 'trainYclass_dualtask.joblib')

    dump(val_X, 'valX_dualtask.joblib')
    dump(val_Ytrend, 'valYtrend_dualtask.joblib')
    dump(val_Yclass, 'valYclass_dualtask.joblib')

    dump(test_X, 'testX_dualtask.joblib')
    dump(test_Ytrend, 'testYtrend_dualtask.joblib')
    dump(test_Yclass, 'testYclass_dualtask.joblib')

    print("✅ 数据转换为 Tensor 并保存完毕")
    
    # 测试数据集中的不同工况占比
    train_labels = train_df['label']
    val_labels = val_df['label']
    test_labels = test_df['label']

    # 打印每个子集的标签分布比例
    print("训练集标签分布:", train_labels.value_counts(normalize=True))
    print("验证集标签分布:", val_labels.value_counts(normalize=True))
    print("测试集标签分布:", test_labels.value_counts(normalize=True))





