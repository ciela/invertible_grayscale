import torch
import torch.nn as nn
import torch.nn.functional as F

import util


class ResidualBlock(nn.Module):

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3)
        self.conv2 = nn.Conv2d(channels, channels, 3)

    def forward(self, x_ini: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x_ini)
        x = F.relu(x)
        x = self.conv2(x)
        return x_ini + x


class ConvTwiceBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(ConvTwiceBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class UpsampleBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(UpsampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)

    def forward(self, x: torch.Tensor, x_res: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + x_res


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.residual64_1 = ResidualBlock(64)
        self.residual64_2 = ResidualBlock(64)
        self.conv_twice128 = ConvTwiceBlock(64, 128)
        self.conv_twice256 = ConvTwiceBlock(128, 256)
        self.residual256_1 = ResidualBlock(256)
        self.residual256_2 = ResidualBlock(256)
        self.residual256_3 = ResidualBlock(256)
        self.residual256_4 = ResidualBlock(256)
        self.upsample128 = UpsampleBlock(256, 128)
        self.upsample64 = UpsampleBlock(128, 64)
        self.residual64_3 = ResidualBlock(64)
        self.residual64_4 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(64, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.residual64_1(x)
        x_res_A = self.residual64_2(x)  # to upsample64
        x_res_B = self.conv_twice128(x_res_A)  # to upsample128
        x = self.conv_twice256(x_res_B)
        x = self.residual256_1(x)
        x = self.residual256_2(x)
        x = self.residual256_3(x)
        x = self.residual256_4(x)
        x = self.upsample128(x, x_res_B)
        x = self.upsample64(x, x_res_A)
        x = self.residual64_3(x)
        x = self.residual64_4(x)
        x = self.conv2(x)
        x = F.tanh(x)
        return x


# TODO: delete
def main():
    encoder = Encoder()
    # pil_img = util.pil_loader('image_path')
    # img_tensor = util.DEFAULT_TRANSFORM(pil_img).unsqueeze(0)
    img_tensor = torch.randn((3, 256, 256)).view(1, 3, 256, 256)
    encoder(img_tensor)


if __name__ == "__main__":
    main()
