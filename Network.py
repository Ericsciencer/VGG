import torch
import torch.nn as nn

# VGG网络配置字典：数字代表卷积层输出通道数，'M'代表最大池化层
VGG_CONFIGS = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        # 根据配置生成卷积层部分
        self.features = self._make_layers(VGG_CONFIGS[vgg_name])
        # 自适应平均池化，将特征图固定为7x7
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # 全连接层分类器
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        # 初始化权重
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # 卷积层前向传播
        x = self.features(x)
        # 自适应池化
        x = self.avgpool(x)
        # 展平特征图
        x = torch.flatten(x, 1)
        # 全连接层分类
        x = self.classifier(x)
        return x

    def _make_layers(self, config):
        """根据配置生成卷积层序列"""
        layers = []
        in_channels = 3  # 输入为RGB图像，通道数为3
        for v in config:
            if v == 'M':
                # 最大池化层：核大小2x2，步长2
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # 卷积层：3x3卷积核，填充1保持尺寸不变
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用Kaiming正态分布初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层使用正态分布初始化，均值0，标准差0.01
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 便捷函数：生成VGG16模型
def vgg16(num_classes=1000, init_weights=True):
    return VGG('vgg16', num_classes=num_classes, init_weights=init_weights)

# 便捷函数：生成VGG19模型
def vgg19(num_classes=1000, init_weights=True):
    return VGG('vgg19', num_classes=num_classes, init_weights=init_weights)


# ---------------- 测试代码 ----------------
if __name__ == "__main__":
    # 创建VGG16模型
    model = vgg16(num_classes=1000)
    # 打印模型结构
    print(model)
    
    # 测试前向传播：输入为224x224的RGB图像，batch size为2
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")  # 应为(2, 1000)