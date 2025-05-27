import torch
import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.signal as signal
from joblib import load, dump
from sklearn.utils import shuffle
from scipy.signal import butter, filtfilt

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

data_12k_10c = pd.DataFrame()

# 读取数据
for index, file in enumerate(file_names):
    # 构建完整文件路径
    file_path = f"{base_directory}/{file}"
    # 读取MAT文件
    data = loadmat(file_path)
    dataList = data[data_columns[index]].reshape(-1)
    data_12k_10c[columns_name[index]] = dataList[:119808]  # 取前119808个数据点

# 打印数据的形状
print(data_12k_10c.shape)
data_12k_10c

data_12k_10c.set_index('de_normal', inplace=True)
data_12k_10c.to_csv('data_12k_10c.csv')
print(data_12k_10c.shape)

# ------------------------
# 带通滤波器设计（200Hz ~ 5900Hz，假设采样率 12000Hz）
# ------------------------
def bandpass_filter(data, lowcut=200, highcut=5900, fs=12000, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# ------------------------
# 数据归一化
# ------------------------
def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val + 1e-8)

# ------------------------
# 构造样本函数（已更新为双任务学习格式）
# ------------------------
def create_dualtask_samples(data, label, Lhist=1024, Lpred=1024, stride=512):
    """
    ✅ 构造输入输出样本 [Lhist 输入] + [Lpred 趋势预测目标] + 标签
    用于多任务学习（分类 + 趋势预测）
    """
    L = Lhist + Lpred
    samples = []
    for i in range(0, len(data) - L + 1, stride):
        x = data[i: i + Lhist]
        y = data[i + Lhist: i + L]
        samples.append(np.concatenate([x, y, [label]]))  # 拼接 x、y 和标签
    return np.array(samples)

# ------------------------
# 主函数
# ------------------------
def process_dataset(filename, Lhist=1024, Lpred=1024, stride=512, split_rate=[0.6, 0.2, 0.2]):
    origin_data = pd.read_csv(filename)
    all_samples = []

    label = 0
    for col_name in origin_data.columns:
        col_data = origin_data[col_name].values
        # 滤波
        filtered = bandpass_filter(col_data)
        # 归一化
        normed = normalize(filtered)
        # ✅ 调用双任务样本构建函数
        samples = create_dualtask_samples(normed, label, Lhist, Lpred, stride)
        all_samples.append(samples)
        label += 1

    # 拼接并打乱
    total_samples = np.vstack(all_samples)
    total_samples = shuffle(total_samples, random_state=42)

    # 划分数据集
    N = len(total_samples)
    n_train = int(N * split_rate[0])
    n_val = int(N * split_rate[1])
    train_set = total_samples[:n_train]
    val_set = total_samples[n_train:n_train + n_val]
    test_set = total_samples[n_train + n_val:]

    # 转换为 DataFrame 保存
    col_count = Lhist + Lpred
    columns = [f'feat_{i}' for i in range(col_count)] + ['label']
    train_df = pd.DataFrame(train_set, columns=columns)
    val_df = pd.DataFrame(val_set, columns=columns)
    test_df = pd.DataFrame(test_set, columns=columns)

    # 保存
    dump(train_df, 'train_set')
    dump(val_df, 'val_set')
    dump(test_df, 'test_set')
    print("✅ 数据处理与保存完毕")

    return train_df, val_df, test_df

train_df, val_df, test_df = process_dataset('data_12k_10c.csv')
print("训练集样本数:", len(train_df))
print("验证集样本数:", len(val_df))
print("测试集样本数:", len(test_df))

# ------------------------
# 将 DataFrame 转换为双任务学习所需的输入和输出张量

def make_dualtask_data_labels(dataframe, Lhist=1024, Lpred=1024):
    """
    将 dataframe 转换为双任务学习所需的输入和输出张量

    参数:
        dataframe: 包含样本的 DataFrame，每行格式应为 [Lhist输入, Lpred目标, 类别标签]
        Lhist: 用于输入的历史时间步数
        Lpred: 用于预测的未来时间步数

    返回:
        x_data: 输入序列 (N, Lhist)
        y_trend: 预测目标 (N, Lpred)
        y_class: 分类标签 (N,)
    """
    input_end = Lhist
    trend_end = Lhist + Lpred

    # 从 DataFrame 中分出输入、预测目标和分类标签
    x_data = dataframe.iloc[:, 0:input_end]
    y_trend = dataframe.iloc[:, input_end:trend_end]
    y_class = dataframe.iloc[:, -1]

    # 转换为 Tensor 格式
    x_tensor = torch.tensor(x_data.values).float()
    y_trend_tensor = torch.tensor(y_trend.values).float()
    y_class_tensor = torch.tensor(y_class.values.astype('int64'))

    return x_tensor, y_trend_tensor, y_class_tensor

# 加载保存的 dataframe 格式数据
train_df = load('train_set')
val_df = load('val_set')
test_df = load('test_set')

# 转换为 tensor
train_X, train_Ytrend, train_Yclass = make_dualtask_data_labels(train_df)
val_X, val_Ytrend, val_Yclass = make_dualtask_data_labels(val_df)
test_X, test_Ytrend, test_Yclass = make_dualtask_data_labels(test_df)

# 保存为 .joblib 文件
dump(train_X, 'trainX_dualtask')
dump(train_Ytrend, 'trainYtrend_dualtask')
dump(train_Yclass, 'trainYclass_dualtask')

dump(val_X, 'valX_dualtask')
dump(val_Ytrend, 'valYtrend_dualtask')
dump(val_Yclass, 'valYclass_dualtask')

dump(test_X, 'testX_dualtask')
dump(test_Ytrend, 'testYtrend_dualtask')
dump(test_Yclass, 'testYclass_dualtask')
print("✅ 数据转换为 Tensor 并保存完毕")
print("训练集样本数:", len(train_X))
print("验证集样本数:", len(val_X))
print("测试集样本数:", len(test_X))
print("训练集标签类别数:", len(torch.unique(train_Yclass)))
print("验证集标签类别数:", len(torch.unique(val_Yclass)))
print("测试集标签类别数:", len(torch.unique(test_Yclass)))
# 打印数据集信息
print("训练集输入形状:", train_X.shape)
print("训练集预测目标形状:", train_Ytrend.shape)
print("训练集分类标签形状:", train_Yclass.shape)
print("验证集输入形状:", val_X.shape)
print("验证集预测目标形状:", val_Ytrend.shape)
print("验证集分类标签形状:", val_Yclass.shape)
print("测试集输入形状:", test_X.shape)
print("测试集预测目标形状:", test_Ytrend.shape)
print("测试集分类标签形状:", test_Yclass.shape)
# 打印数据集统计信息
print("训练集输入数据类型:", train_X.dtype)
print("训练集预测目标数据类型:", train_Ytrend.dtype)
print("训练集分类标签数据类型:", train_Yclass.dtype)
print("验证集输入数据类型:", val_X.dtype)
print("验证集预测目标数据类型:", val_Ytrend.dtype)
print("验证集分类标签数据类型:", val_Yclass.dtype)
print("测试集输入数据类型:", test_X.dtype)
print("测试集预测目标数据类型:", test_Ytrend.dtype)
print("测试集分类标签数据类型:", test_Yclass.dtype)
print("✅ 数据集处理完成")
print("训练集样本数:", len(train_X))
print("验证集样本数:", len(val_X))
print("测试集样本数:", len(test_X))
print("训练集标签类别数:", len(torch.unique(train_Yclass)))
print("验证集标签类别数:", len(torch.unique(val_Yclass)))
print("测试集标签类别数:", len(torch.unique(test_Yclass)))


