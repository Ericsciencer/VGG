import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ----------------------
# 1. VGG16-CIFAR 精简版模型定义（保留VGG核心思想，适配32x32输入）
# ----------------------
class VGG16_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_CIFAR, self).__init__()
        # 核心思想：3x3卷积堆叠 + 通道数逐层翻倍 + 池化降维
        self.features = nn.Sequential(
            # Block 1 (输入: 3x32x32)
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 64x16x16
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 128x8x8
            
            # Block 3 (模拟VGG16的深层堆叠)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 256x4x4
        )
        
        # 精简版全连接层（适配CIFAR-10的10分类）
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
        
        # 权重初始化（符合VGG论文实践）
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 4 * 4)  # 展平
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# ----------------------
# 2. 数据加载与预处理（适配CIFAR-10的32x32输入）
# ----------------------
def get_data_loaders(batch_size=64):
    # CIFAR-10专用归一化参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

# ----------------------
# 3. 训练函数
# ----------------------
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item() * images.size(0)
    
    avg_train_loss = total_loss / len(train_loader.dataset)
    avg_train_acc = correct / total
    return avg_train_loss, avg_train_acc

# ----------------------
# 4. 测试函数
# ----------------------
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ----------------------
# 5. 主程序
# ----------------------
if __name__ == '__main__':
    # 基础配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    lr = 0.01
    num_epochs = 15

    # 初始化核心组件
    model = VGG16_CIFAR(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # VGG论文优化器
    train_loader, test_loader = get_data_loaders(batch_size)

    # 初始化列表存储指标
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 训练循环
    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_acc = test(model, test_loader, device)
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # 保存模型权重
    torch.save(model.state_dict(), 'vgg16_cifar10.pth')
    print("Model saved as vgg16_cifar10.pth")

    # 可视化绘图
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, train_loss_list, 'b-', linewidth=2, label='train loss')
    plt.plot(epochs, train_acc_list, 'm--', linewidth=2, label='train acc')
    plt.plot(epochs, test_acc_list, 'g--', linewidth=2, label='test acc')
    plt.xlabel('epoch', fontsize=18)
    plt.xticks(range(2, 11, 2))
    plt.ylim(0, 2.4)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=18)
    plt.title('VGG16-CIFAR Training Metrics', fontsize=16)
    plt.savefig('vgg16_cifar_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()