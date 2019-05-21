import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):

    def __init__(self, weights: tuple, gc_weights: tuple):
        super(Loss, self).__init__()
        self.w = weights
        self.gcw = gc_weights

    def forward(self, orig: torch.Tensor, grayscale: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
        interbility = self.w[0] * Loss.invertibility(orig, restored)
        grayscale_conformity = self.w[1] * Loss.grayscale_conformity(orig, grayscale, self.gcw)
        quantization = self.w[2] * Loss.quantization(grayscale)
        full_loss = interbility + grayscale_conformity + quantization
        return full_loss

    @staticmethod
    def invertibility(orig: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(orig, restored)

    @staticmethod
    def grayscale_conformity(orig: torch.Tensor, grayscale: torch.Tensor, weights: tuple) -> torch.Tensor:
        lightness = weights[0] * Loss.gc_lightness(orig, grayscale)
        contrast = weights[1] * Loss.gc_contrast(orig, grayscale)
        local_structure = weights[2] * Loss.gc_local_structure(orig, grayscale)
        return lightness + contrast + local_structure

    @staticmethod
    def gc_lightness(orig: torch.Tensor, grayscale: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(orig, orig)

    @staticmethod
    def gc_contrast(orig: torch.Tensor, grayscale: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(orig, orig)

    @staticmethod
    def gc_local_structure(orig: torch.Tensor, grayscale: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(orig, orig)

    @staticmethod
    def quantization(grayscale: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(grayscale, grayscale)
