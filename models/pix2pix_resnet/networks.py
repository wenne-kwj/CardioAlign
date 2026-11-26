"""ResNet-50 encoder-decoder and adaptive conditional discriminator, from the supplied implementation."""
import torch
from torch import nn
from torch.nn import functional as F

def _norm(num_features):
    """Normalization that is stable for the high-resolution batch size of 1-2."""
    return nn.InstanceNorm2d(num_features, affine=True, track_running_stats=False)


class ResNetBottleneck(nn.Module):
    """标准 ResNet Bottleneck 模块"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(ResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = _norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = _norm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = _norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class AdaptiveGenerator(nn.Module):
    """
    基于 ResNet-50 的自适应生成器
    结构：ResNet-50 Encoder -> ResNet-style Decoder
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(AdaptiveGenerator, self).__init__()
        self.inplanes = ngf

        # --- Encoder: ResNet-50 部分 ---
        self.initial = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=2, padding=3, bias=False),
            _norm(ngf),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet 层级配置 [3, 4, 6, 3]
        self.layer1 = self._make_layer(ngf, 3)
        self.layer2 = self._make_layer(ngf * 2, 4, stride=2)
        self.layer3 = self._make_layer(ngf * 4, 6, stride=2)
        self.layer4 = self._make_layer(ngf * 8, 3, stride=2)

        # --- Decoder: 上采样部分 ---
        # 2048 -> 1024 -> 512 -> 256 -> 128 -> 64
        self.up1 = self._make_up_layer(ngf * 32, ngf * 16)
        self.up2 = self._make_up_layer(ngf * 32, ngf * 8)
        self.up3 = self._make_up_layer(ngf * 16, ngf * 4)
        self.up4 = self._make_up_layer(ngf * 8, ngf * 2)
        self.up5 = self._make_up_layer(ngf * 3, ngf)

        # 最终输出层
        self.final = nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * ResNetBottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * ResNetBottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                _norm(planes * ResNetBottleneck.expansion),
            )
        layers = [ResNetBottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * ResNetBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(ResNetBottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_up_layer(self, in_nc, out_nc):
        return nn.Sequential(
            nn.Conv2d(in_nc, out_nc, kernel_size=3, padding=1),
            _norm(out_nc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        original_size = x.shape[2:]

        # Encoder
        x1 = self.initial(x)      # /4
        x2 = self.layer1(x1)      # /4
        x3 = self.layer2(x2)      # /8
        x4 = self.layer3(x3)      # /16
        x5 = self.layer4(x4)      # /32

        # Decoder (逐级上采样并检查尺寸)
        # 注意：这里为了保持 ResNet-50 的深度，采用了标准的 Decoder 流程
        def upsample_block(feat, target_size, up_module):
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            return up_module(feat)

        u1 = torch.cat([upsample_block(x5, x4.shape[2:], self.up1), x4], dim=1)
        u2 = torch.cat([upsample_block(u1, x3.shape[2:], self.up2), x3], dim=1)
        u3 = torch.cat([upsample_block(u2, x2.shape[2:], self.up3), x2], dim=1)
        u4 = torch.cat([upsample_block(u3, x1.shape[2:], self.up4), x1], dim=1)
        u5 = upsample_block(u4, original_size, self.up5)

        output = self.final(u5)

        # 最终强制匹配（防止奇数尺寸导致的 1 像素偏差）
        if output.shape[2:] != original_size:
            output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)

        return output


class AdaptiveDiscriminator(nn.Module):
    """自适应判别器"""
    def __init__(self, input_nc=6, ndf=64):
        super(AdaptiveDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(_norm(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_nc, ndf, False),
            *discriminator_block(ndf, ndf * 2),
            *discriminator_block(ndf * 2, ndf * 4),
            *discriminator_block(ndf * 4, ndf * 8),
        )
        self.classifier = nn.Conv2d(ndf * 8, 1, 4, padding=1, bias=False)

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        x = img_input
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and min(x.shape[2:]) <= 8:
                break
        return self.classifier(x)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
