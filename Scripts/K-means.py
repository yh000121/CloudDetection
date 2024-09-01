import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import glob
from sklearn.decomposition import PCA

def load_features(feature_file):
    # 加载特征数据
    features_data = np.load(feature_file)

    # 再次检查并处理 NaN 值
    if np.isnan(features_data).any():
        print("Warning: NaN values found in features during K-means clustering. Replacing NaNs with 0.")
        features_data = np.nan_to_num(features_data, nan=0.0)

    return features_data


def kmeans_clustering(features, num_clusters=3, n_components=2):
    rows, cols, num_layers = features.shape
    reshaped_features = features.reshape(rows * cols, num_layers)

    # 使用 PCA 降维
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(reshaped_features)

    # 执行 K-means 聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(reduced_features)
    cluster_labels = kmeans.labels_

    # 将聚类标签重塑回原始图像的维度
    clustered_image = cluster_labels.reshape(rows, cols)

    return clustered_image



def save_clustered_image(clustered_image, output_filename):
    # 清除当前图形，以防止多次绘制时叠加 colorbar
    plt.figure()

    # 绘制聚类结果，并设置图像比例
    plt.imshow(clustered_image, cmap='viridis', aspect='auto')  # 可以试试 'equal' 或 'auto'

    # 设置 x 和 y 轴的刻度
    plt.xticks(ticks=range(0, clustered_image.shape[1], 200))
    plt.yticks(ticks=range(0, clustered_image.shape[0], 200))

    # 添加 colorbar
    plt.colorbar()

    # 保存图像并关闭图形窗口
    plt.savefig(output_filename)
    plt.close()  # 确保关闭图形窗口以防止后续绘制时叠加


if __name__ == "__main__":
    # 假设处理后的数据文件位于以下目录
    feature_dir = 'processed_data/features'
    output_dir = '../Results/clustered_images'  # 调整路径，使其指向 CloudDetection/Results 文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有处理过的特征文件
    feature_files = sorted(glob.glob(os.path.join(feature_dir, '*_reshaped_features.npy')))

    for feature_file in feature_files:
        # 获取子目录名（如 171）
        subdir = os.path.basename(feature_file).replace('_reshaped_features.npy', '')

        # 加载特征数据
        features = load_features(feature_file)

        # 执行 K-means 聚类
        clustered_image = kmeans_clustering(features, num_clusters=3)
        print(f"K-means clustering completed for {subdir}.")

        # 保存聚类结果为图像
        clustered_image_path = os.path.join(output_dir, f"{subdir}_clustered.png")
        save_clustered_image(clustered_image, clustered_image_path)
        print(f"Clustered image saved to {clustered_image_path}.")

