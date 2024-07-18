import glob
import os
import numpy as np
import xarray as xr
import torch

def preprocess_radiance_data(ds):
    # 提取所有 radiance 图层
    radiance_layers = [key for key in ds.data_vars.keys() if 'radiance_in' in key]

    # 获取图像的维度
    rows, cols = ds.sizes['rows'], ds.sizes['columns']

    # 创建一个存储所有图层数据的多维数组
    num_layers = len(radiance_layers)
    all_features = np.zeros((rows, cols, num_layers), dtype=np.float32)

    for i, layer in enumerate(radiance_layers):
        # 提取数据并计算 Dask 数组
        data = ds[layer].data.compute()

        # 确认数据形状
        print(f"Layer {layer} data shape: {data.shape}")

        # 转换为 PyTorch Tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # 检查并替换 NaN 和 Inf 值
        if torch.isnan(data_tensor).any() or torch.isinf(data_tensor).any():
            print(f"Data in layer {layer} contains NaN or Inf values. Replacing with 0.")
            data_tensor = torch.nan_to_num(data_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        # 标准化
        data_standardized = (data_tensor - data_tensor.mean()) / data_tensor.std()

        # 归一化
        data_normalized = (data_standardized - data_standardized.min()) / (data_standardized.max() - data_standardized.min())

        # 将标准化后的数据存储到对应的特征位置
        all_features[:, :, i] = data_normalized.numpy()

    return all_features, rows, cols

def load_single_label(label_path):
    # 读取标签数据
    ds_label = xr.open_dataset(label_path)

    # 获取标签数据数组
    label_key = list(ds_label.data_vars.keys())[0]  # 获取第一个变量名
    label_data = ds_label[label_key].data

    return label_data

def load_label_data(label_paths, rows, cols):
    # 创建一个空的标签数组，初始值为0
    labels_combined = np.zeros((rows, cols), dtype=int)

    for idx, label_path in enumerate(label_paths):
        # 加载单个标签文件
        label_data = load_single_label(label_path)

        # 打印每个标签文件的形状
        print(f"Label data shape for {label_path}: {label_data.shape}")

        # 确认标签数据的形状与预处理数据的形状一致
        if label_data.shape != (rows, cols):
            raise ValueError(f"标签数据的形状与预处理数据不一致: {label_path}")

        # 将标签数据中的非零值设置为类别索引
        labels_combined[label_data == 1] = idx + 1  # 1: ice, 2: clear, 3: cloud

    return labels_combined

if __name__ == "__main__":
    base_path = '../images'

    # 获取所有子目录
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # 创建文件夹以保存预处理后的数据
    os.makedirs('processed_data', exist_ok=True)

    # 创建一个空的列表来存储所有样本的特征和标签
    all_samples = []
    all_labels = []

    for subdir in subdirs:
        subdir_path = os.path.join(base_path, subdir)
        # 读取多个 NetCDF 文件并合并到一个 Dataset 中
        ds = xr.open_mfdataset(f'{subdir_path}/S*_radiance_in.nc', combine='by_coords')

        # 预处理辐射层数据
        all_features, rows, cols = preprocess_radiance_data(ds)
        print(f"All features shape for {subdir}: {all_features.shape}")

        # 增加样本维度并存储到列表
        all_samples.append(all_features)

        # 检查是否有标签文件
        label_paths = [os.path.join(subdir_path, label) for label in ['ice_labels.nc', 'clear_labels.nc', 'cloud_labels.nc']]
        if all(os.path.exists(label_path) for label_path in label_paths):
            labels = load_label_data(label_paths, rows, cols)
            all_labels.append(labels)

    # 合并所有样本 (增加一个维度用于区分样本)
    combined_features = np.stack(all_samples)
    print(f"Combined features shape: {combined_features.shape}")

    # 保存合并后的辐射层数据
    np.save("processed_data/preprocessed_data.npy", combined_features)

    if all_labels:
        # 合并所有标签数据
        combined_labels = np.stack(all_labels)
        print(f"Combined labels shape: {combined_labels.shape}")
        np.save("processed_data/labels.npy", combined_labels)
