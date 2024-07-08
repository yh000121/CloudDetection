import torch
import numpy as np
import xarray as xr

def compute_mean_std(tensor):
    mean = tensor.mean().item()
    std = tensor.std().item()
    return mean, std

def standardize(tensor):
    mean, std = compute_mean_std(tensor)
    return (tensor - mean) / std

def preprocess_data(ds):
    # 提取所有 radiance 图层
    radiance_layers = [key for key in ds.data_vars.keys() if 'radiance_an' in key]

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

        # 标准化
        data_standardized = standardize(data_tensor)

        # 将标准化后的数据存储到对应的特征位置
        all_features[:, :, i] = data_standardized.numpy()

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

        # 确认标签数据的形状与预处理数据的形状一致
        if label_data.shape != (rows, cols):
            raise ValueError(f"标签数据的形状与预处理数据不一致: {label_path}")

        # 将标签数据中的非零值设置为类别索引
        labels_combined[label_data == 1] = idx + 1  # 1: ice, 2: clear, 3: cloud

    return labels_combined


# 示例调用
if __name__ == "__main__":
    # 定义数据文件路径
    path = '../images/S1'

    # 读取多个 NetCDF 文件并合并到一个 Dataset 中
    ds = xr.open_mfdataset(f'{path}/S*_radiance_an.nc', combine='by_coords')

    # 预处理数据
    all_features, rows, cols = preprocess_data(ds)

    # 打印结果以检查
    print(f"All features shape: {all_features.shape}")
    print(f"Rows: {rows}, Columns: {cols}")


    # 加载标签数据
    label_paths = [
        '../images/S1/ice_labels.nc',
        '../images/S1/clear_labels.nc',
        '../images/S1/cloud_labels.nc'
    ]
    label_data = load_label_data(label_paths, rows, cols)

    print(f"Combined labels shape: {label_data.shape}")
