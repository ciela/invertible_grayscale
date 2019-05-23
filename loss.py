import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models.vgg import vgg19


VGG_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # model-zoo manner
])


class Loss(nn.Module):

    def __init__(self,
        weights1: tuple = (1.0, 0.0), weights2: tuple = (0.5, 10.0),
        gc_weights: tuple = (1e-7, 0.5), gc_lightness_theta: int = 70):
        super(Loss, self).__init__()
        self.w1 = weights1
        self.w2 = weights2
        self.gcw = gc_weights
        self.theta = gc_lightness_theta
        self.vgg = vgg19(pretrained=True)
        self.vgg_conv4_4_feat = torch.zeros((1, 512, 28, 28))
        def vgg_conv4_4_hook(m: nn.Module, i: tuple, o: torch.Tensor):
            self.vgg_conv4_4_feat.copy_(o)  # copies with grad_fn
        self.vgg.features[25].register_forward_hook(vgg_conv4_4_hook)  # conv4_4
        self.vgg.eval()

    def forward(self, X_orig_color: torch.Tensor, T_orig_gray: torch.Tensor,
        Y_grayscale: torch.Tensor, Y_restored: torch.Tensor, stage: int = 1) -> torch.Tensor:
        weights = self.w1 if stage == 1 else self.w2
        invertibility = self.invertibility(X_orig_color, Y_restored)
        grayscale_conformity = weights[0] * self.grayscale_conformity(T_orig_gray, Y_grayscale)
        quantization = weights[1] * self.quantization(Y_grayscale)
        full_loss = invertibility + grayscale_conformity + quantization
        print('Full Loss: ', invertibility, grayscale_conformity, quantization)
        return full_loss

    def invertibility(self, X_orig_color: torch.Tensor, Y_restored: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(X_orig_color, Y_restored)

    def grayscale_conformity(self, T_orig_gray: torch.Tensor, Y_grayscale: torch.Tensor) -> torch.Tensor:
        lightness = self.gc_lightness(T_orig_gray, Y_grayscale)
        contrast = self.gcw[0] * self.gc_contrast(T_orig_gray, Y_grayscale)
        local_structure = self.gcw[1] * self.gc_local_structure(T_orig_gray, Y_grayscale)
        print('GC Loss: ', lightness, contrast, local_structure)
        return lightness + contrast + local_structure

    def gc_lightness(self, T_orig_gray: torch.Tensor, Y_grayscale: torch.Tensor) -> torch.Tensor:
        # abs range is [0, 2]
        return torch.mean(
            torch.max(
                torch.abs(Y_grayscale - T_orig_gray) - self.theta / 127,
                torch.zeros_like(T_orig_gray)))

    def gc_contrast(self, T_orig_gray: torch.Tensor, Y_grayscale: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.vgg(VGG_TRANSFORM(T_orig_gray.squeeze(0)).unsqueeze(0))
            T_orig_gray = self.vgg_conv4_4_feat.clone()
        self.vgg(VGG_TRANSFORM(Y_grayscale.squeeze(0)).unsqueeze(0))
        Y_grayscale = self.vgg_conv4_4_feat.clone()
        return F.l1_loss(T_orig_gray, Y_grayscale)

    def gc_local_structure(self, T_orig_gray: torch.Tensor, Y_grayscale: torch.Tensor) -> torch.Tensor:
        # TODO: Calculate diffs of Total-Variations
        return F.mse_loss(T_orig_gray, T_orig_gray)

    def quantization(self, Y_grayscale: torch.Tensor) -> torch.Tensor:
        grayscale_stack = torch.stack([Y_grayscale for _ in range(256)])
        M = torch.zeros_like(Y_grayscale)
        M_stack = torch.stack([M.new_full(M.size(), fill_value=d) for d in range(256)])
        M_stack = (M_stack / 127.5) - 1
        return torch.mean(
            torch.min(
                torch.min(
                    torch.abs(grayscale_stack - M_stack), dim=3).values, dim=2).values)
