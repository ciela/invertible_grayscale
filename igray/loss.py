import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models.vgg import vgg19


def to_vgg_ready(t: torch.Tensor) -> torch.Tensor:
    t = t.squeeze(0)  # (1, 1, :, :) -> (1, :, :)
    t = torch.cat([t for _ in range(3)])  # (1, :, :) -> (3, :, :)
    t = t.unsqueeze(0)  # (3, :, :) -> (1, 3, :, :)
    t = t + 1 / 2  # tentative normalization for model-zoo
    return t


class Loss(nn.Module):

    def __init__(self,
        weights1: tuple = (1.0, 0.0), weights2: tuple = (0.5, 10.0),
        gc_weights: tuple = (1e-7, 0.5), gc_lightness_theta: int = 70):
        super(Loss, self).__init__()
        self.w1 = weights1
        self.w2 = weights2
        self.gcw = gc_weights
        self.theta = gc_lightness_theta
        self.vgg_conv44 = vgg19(pretrained=True).features[:27]  # conv4_4 + relu
        self.vgg_conv44.eval()

    def forward(self, X_orig_color: torch.Tensor, T_orig_gray: torch.Tensor,
        Y_grayscale: torch.Tensor, Y_restored: torch.Tensor, stage: int = 1) -> torch.Tensor:
        weights = self.w1 if stage == 1 else self.w2
        invertibility = 3 * self.invertibility(X_orig_color, Y_restored)
        grayscale_conformity = weights[0] * self.grayscale_conformity(T_orig_gray, Y_grayscale)
        full_loss = invertibility + grayscale_conformity
        if stage == 2:
            quantization = weights[1] * self.quantization(Y_grayscale)
            full_loss += quantization
        return full_loss

    def invertibility(self, X_orig_color: torch.Tensor, Y_restored: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(X_orig_color, Y_restored)

    def grayscale_conformity(self, T_orig_gray: torch.Tensor, Y_grayscale: torch.Tensor) -> torch.Tensor:
        lightness = self.gc_lightness(T_orig_gray, Y_grayscale)
        contrast = self.gcw[0] * self.gc_contrast(T_orig_gray, Y_grayscale)
        local_structure = self.gcw[1] * self.gc_local_structure(T_orig_gray, Y_grayscale)
        return lightness + contrast + local_structure

    def gc_lightness(self, T_orig_gray: torch.Tensor, Y_grayscale: torch.Tensor) -> torch.Tensor:
        # abs range is [0, 2]
        return torch.mean(
            torch.max(
                torch.abs(Y_grayscale - T_orig_gray) - self.theta / 127,
                torch.zeros_like(T_orig_gray)))

    def gc_contrast(self, T_orig_gray: torch.Tensor, Y_grayscale: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            T_orig_gray = self.vgg_conv44(to_vgg_ready(T_orig_gray))
        Y_grayscale = self.vgg_conv44(to_vgg_ready(Y_grayscale))
        return F.l1_loss(T_orig_gray, Y_grayscale)

    def gc_local_structure(self, T: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        T_tv = torch.sum(torch.abs(T[:, :, :-1, :] - T[:, :, 1:, :]))\
            + torch.sum(torch.abs(T[:, :, :, :-1] - T[:, :, :, 1:]))
        Y_tv = torch.sum(torch.abs(Y[:, :, :-1, :] - Y[:, :, 1:, :]))\
            + torch.sum(torch.abs(Y[:, :, :, :-1] - Y[:, :, :, 1:]))
        return F.l1_loss(T_tv / 256 ** 2, Y_tv / 256 ** 2)

    def quantization(self, Y_grayscale: torch.Tensor) -> torch.Tensor:
        grayscale_stack = torch.cat([Y_grayscale for _ in range(256)])
        M = torch.zeros_like(Y_grayscale)
        M_stack = torch.cat([M.new_full(M.size(), fill_value=d) for d in range(256)])
        M_stack = (M_stack / 127.5) - 1
        absmin = torch.min(
            torch.min(
                torch.abs(grayscale_stack - M_stack), dim=3).values, dim=2).values
        result = torch.mean(absmin)
        return result
