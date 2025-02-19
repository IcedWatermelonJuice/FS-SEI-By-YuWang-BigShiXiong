import numpy as np
from sklearn.model_selection import train_test_split
import random
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import sys


def add_noise(x, snr=20):
    """
        为信号添加噪声
        :param x: 待处理信号x, 形状为 batch_size, 2, signal_length
        :param snr: 信噪比
        :return: 添加噪声后的信号x_noisy
        """
    # 获取信号长度
    signal_length = x.shape[2]
    # 计算信号的功率
    signal_power = np.sum(np.power(x, 2), axis=2) / signal_length
    # 计算噪声功率
    noise_power = signal_power / (10 ** (snr / 10))
    # 生成与输入数组形状相同的高斯噪声
    noise = np.random.normal(size=x.shape)
    # 计算当前噪声功率
    current_noise_power = np.sum(np.power(noise, 2), axis=2) / signal_length
    # 根据噪声功率对噪声进行缩放
    noise = noise * np.sqrt(noise_power / current_noise_power)[..., np.newaxis]
    # 将噪声添加到输入数组
    x_noisy = x + noise
    return x_noisy


def power_normalize_fn(x):
    for i in range(x.shape[0]):
        max_power = (np.power(x[i, 0, :], 2) + np.power(x[i, 1, :], 2)).max()
        x[i] = x[i] / np.power(max_power, 1 / 2)
    return x


def maxmin_normalize_fn(x):
    min_value = x.min()
    max_value = x.max()
    x = (x - min_value) / (max_value - min_value)
    return x


def normalize_dataX(x):
    return maxmin_normalize_fn(x)
    # return power_normalize_fn(x)


def get_dataset_info(dataset_root, num_class, is_pt=True):
    def_info = {
        "linux": "~/Datasets/ADS-B",
        "windows": "E:\\Datasets\\ADS-B",
        "pt_class": 90,
        "ft_class": 30
    }
    platform = "windows" if sys.platform.startswith("win") else "linux"
    dataset_root = def_info[platform] if dataset_root == "default" else dataset_root
    num_class = def_info["pt_class" if is_pt else "ft_class"] if num_class == "default" else num_class
    return dataset_root, num_class


def load_data(dataset_root, num_class, suffix, ch_type=None):
    ch_type = f"_{ch_type}" if ch_type else ""
    x = np.load(os.path.expanduser(os.path.join(dataset_root, f"X_{suffix}_{num_class}Class{ch_type}.npy")))
    y = np.load(os.path.expanduser(os.path.join(dataset_root, f"Y_{suffix}_{num_class}Class.npy")))

    if len(x.shape) == 3 and x.shape[1] != 2:
        x = x.transpose((0, 2, 1))

    return x[:, :, :4800], y


def pt_train_data(dataset_root, num_class):
    x, y = load_data(dataset_root, num_class, "train")
    x = normalize_dataX(x)
    y = y.astype(np.uint8)

    return x, y


def ft_train_data(random_seed, dataset_root, num_class, k_shot):
    x, y = load_data(dataset_root, num_class, "train")
    if len(x.shape) == 5:
        x = x[:, 0, :, :, :]

    train_index_shot = []
    random.seed(random_seed)
    for i in range(num_class):
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += random.sample(index_classi, k_shot)
    x = x[train_index_shot]
    y = y[train_index_shot]

    x = normalize_dataX(x)
    y = y.astype(np.uint8)
    return x, y


def ft_test_data(dataset_root, num_class):
    x, y = load_data(dataset_root, num_class, "test")
    if len(x.shape) == 5:
        x = x[:, 0, :, :, :]

    x = normalize_dataX(x)
    y = y.astype(np.uint8)

    return x, y


def get_pretrain_dataloader(data_root="default", num_class="default", batch_size=32, random_seed=2024):
    data_root, num_class = get_dataset_info(data_root, num_class, is_pt=True)
    X_train, Y_train = pt_train_data(data_root, num_class)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=random_seed)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader


def get_finetune_dataloader(data_root="default", num_class="default", train_batch_size=32, test_batch_size=30, shot=1, snr=None, random_seed=2024):
    data_root, num_class = get_dataset_info(data_root, num_class, is_pt=False)
    X_train, Y_train = ft_train_data(random_seed, data_root, num_class, shot)
    X_test, Y_test = ft_test_data(data_root, num_class)
    if snr is not None:
        X_test = add_noise(X_test, snr=snr)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).float(), torch.tensor(Y_test))

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    td, vd = get_pretrain_dataloader()

    print("end")
