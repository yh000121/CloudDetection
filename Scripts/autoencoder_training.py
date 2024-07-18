import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_channels):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder(features, input_channels, device):
    # 准备数据
    X = torch.tensor(features, dtype=torch.float32)
    if torch.isnan(X).any() or torch.isinf(X).any():
        print(f"Data contains NaN or Inf values. Replacing with 0.")
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Data stats: min={X.min().item()}, max={X.max().item()}, mean={X.mean().item()}, std={X.std().item()}")
    X = X.permute(0, 3, 1, 2)  # 调整维度为 (num_samples, channels, height, width)
    train_dataset = TensorDataset(X)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # 增加批次大小

    # 定义模型
    autoencoder = SimpleAutoencoder(input_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4)  # 提高学习率

    # 训练模型
    for epoch in range(50):  # 增加训练周期数
        total_loss = 0
        for data in train_loader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            # 在计算损失前确保输出和输入大小一致
            if outputs.size() != inputs.size():
                outputs = nn.functional.interpolate(outputs, size=inputs.size()[2:])
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

    # 保存模型
    model_path = os.path.join("models", "simple_autoencoder.pth")
    os.makedirs("models", exist_ok=True)
    torch.save(autoencoder.state_dict(), model_path)
    print(f"Autoencoder model saved at {model_path}.")
    return autoencoder

if __name__ == "__main__":
    processed_data_path = 'processed_data'

    # 加载预处理后的数据
    combined_features = np.load(os.path.join(processed_data_path, "preprocessed_data.npy"))
    print(f"Combined features shape: {combined_features.shape}")

    # 训练自动编码器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_channels = combined_features.shape[3]  # 确保输入通道数正确
    autoencoder = train_autoencoder(combined_features, input_channels, device)
