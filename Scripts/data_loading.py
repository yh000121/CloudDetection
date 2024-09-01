import glob
import os
import numpy as np
import xarray as xr
import torch

def preprocess_all_layers(ds):
    # 提取 radiance 和 BT 图层
    radiance_layers = [key for key in ds.data_vars.keys() if 'radiance_in' in key]
    bt_layers = [key for key in ds.data_vars.keys() if 'BT_in' in key]

    # 合并 radiance 和 BT 图层列表
    all_layers = radiance_layers + bt_layers

    # 获取图像的维度
    rows, cols = ds.sizes['rows'], ds.sizes['columns']

    # 创建一个存储所有图层数据的多维数组
    num_layers = len(all_layers)
    all_features = np.zeros((rows, cols, num_layers), dtype=np.float32)

    for i, layer in enumerate(all_layers):
        # 提取数据并计算 Dask 数组
        data = ds[layer].data.compute()

        # 转换为 PyTorch Tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # 检查并替换 NaN 值
        if torch.isnan(data_tensor).any():
            print(f"Data in layer {layer} contains NaN values. Replacing with 0.")
            data_tensor = torch.nan_to_num(data_tensor, nan=0.0)

        # 标准化
        data_standardized = (data_tensor - data_tensor.mean()) / data_tensor.std()

        # 归一化
        data_normalized = (data_standardized - data_standardized.min()) / (
                    data_standardized.max() - data_standardized.min())

        # 将标准化后的数据存储到对应的特征位置
        all_features[:, :, i] = data_normalized.numpy()

    return all_features, rows, cols


def preprocess_all_labels(subdir_path):
    # 定义标签文件的路径
    label_files = {
        'clear': f'{subdir_path}/clear_labels.nc',
        'ice': f'{subdir_path}/ice_labels.nc',
        'cloud': f'{subdir_path}/cloud_labels.nc'
    }

    # 初始化存储标签的数组
    combined_labels = None
    rows, cols = None, None

    for i, (label_name, file_path) in enumerate(label_files.items()):
        if os.path.exists(file_path):
            # 加载标签数据
            labels_ds = xr.open_dataset(file_path)

            # 假设每个标签文件只有一个数据变量
            label_layer = list(labels_ds.data_vars.keys())[0]
            label_data = labels_ds[label_layer].data  # 已经是 NumPy 数组，不需要 compute()

            # 初始化 combined_labels 数组
            if combined_labels is None:
                rows, cols = label_data.shape
                combined_labels = np.zeros((rows, cols), dtype=np.int32)

            # 给每个标签分配一个唯一的值
            combined_labels[label_data > 0] = i + 1

        else:
            print(f"Warning: {label_name} labels file not found in {subdir_path}")

    if combined_labels is None:
        raise FileNotFoundError("No valid labels files found in the directory.")

    return combined_labels, rows, cols


if __name__ == "__main__":
    base_path = '../images'

    # 获取所有子目录
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # 创建文件夹以保存预处理后的数据
    output_features_dir = 'processed_data/features'
    output_labels_dir = 'processed_data/labels'
    os.makedirs(output_features_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    for subdir in subdirs:
        subdir_path = os.path.join(base_path, subdir)

        # 读取多个 NetCDF 文件并合并到一个 Dataset 中
        radiance_ds = xr.open_mfdataset(f'{subdir_path}/S*_radiance_in.nc', combine='by_coords')
        bt_ds = xr.open_mfdataset(f'{subdir_path}/S*_BT_in.nc', combine='by_coords')
        ds = xr.merge([radiance_ds, bt_ds])

        # 预处理所有图层数据 (包括 radiance 和 BT)
        all_features, rows, cols = preprocess_all_layers(ds)
        print(f"All features shape for {subdir}: {all_features.shape}")

        # 保存每个子目录的特征数据为单独的文件
        np.save(os.path.join(output_features_dir, f"{subdir}_reshaped_features.npy"), all_features)

        try:
            # 预处理所有labels图层数据
            combined_labels, rows, cols = preprocess_all_labels(subdir_path)
            print(f"All labels shape for {subdir}: {combined_labels.shape}")

            # 保存每个子目录的标签数据为单独的文件
            np.save(os.path.join(output_labels_dir, f"{subdir}_reshaped_labels.npy"), combined_labels)

        except FileNotFoundError:
            # 打印信息并继续处理下一个子目录
            print(f"Skipping {subdir} due to missing label files.")
            continue






