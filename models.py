import torch
import torch.nn as nn
import torch.nn.functional as F

import util


class ResidualBlock(nn.Module):

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x_ini: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x_ini)
        x = F.relu(x)
        x = self.conv2(x)
        return x_ini + x


class DownsampleBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(DownsampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class UpsampleBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(UpsampleBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, x_res: torch.Tensor) -> torch.Tensor:
        x = self.deconv1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        return x + x_res


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.residuals64_1 = nn.ModuleList([ResidualBlock(64) for _ in range(2)])
        self.downsample128 = DownsampleBlock(64, 128)
        self.downsample256 = DownsampleBlock(128, 256)
        self.residuals256 = nn.ModuleList([ResidualBlock(256) for _ in range(4)])
        self.upsample128 = UpsampleBlock(256, 128)
        self.upsample64 = UpsampleBlock(128, 64)
        self.residuals64_2 = nn.ModuleList([ResidualBlock(64) for _ in range(2)])
        self.conv2 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        for residual in self.residuals64_1:
            x = residual(x)
        x_res_A = x  # to upsample64
        x_res_B = self.downsample128(x_res_A)  # to upsample128
        x = self.downsample256(x_res_B)
        for residual in self.residuals256:
            x = residual(x)
        x = self.upsample128(x, x_res_B)  # from downsample128
        x = self.upsample64(x, x_res_A)  # from downsample64
        for residual in self.residuals64_2:
            x = residual(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        return x


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.residuals = nn.ModuleList([ResidualBlock(64) for _ in range(8)])
        self.conv2 = nn.Conv2d(64, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for residual in self.residuals:
            x = residual(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.tanh(x)


# TODO: delete
def main():
    encoder = Encoder()
    print(encoder)
    # pil_img = util.pil_loader('image_path')
    # img_tensor = util.DEFAULT_TRANSFORM(pil_img).unsqueeze(0)
    img_tensor = torch.randn((3, 256, 256)).view(1, 3, 256, 256)
    grayscale = encoder(img_tensor)
    print(grayscale)
    decoder = Decoder()
    print(decoder)
    resotred = decoder(grayscale)
    print(resotred)


if __name__ == "__main__":
    main()
