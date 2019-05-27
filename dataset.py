from typing import Tuple
from pathlib import Path

import PIL.Image as PILImage
from PIL.Image import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # normalize to [-1, 1]
])


GRAYSCALE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),  # grayscale manner
])


def pil_loader(img_path: str) -> Image:
    with open(img_path, 'rb') as f:
        img = PILImage.open(f)
        return img.convert('RGB')


class Dataset(data.Dataset):

    def __init__(self, datadir, num_samples=-1):
        super(Dataset, self).__init__()
        self.datadir = datadir
        self.paths = [str(p) for p in Path(self.datadir).glob('*')]
        if num_samples != -1:
            # TODO: sample
            pass

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.paths[index]
        pil_img = pil_loader(item)
        return DEFAULT_TRANSFORM(pil_img), GRAYSCALE_TRANSFORM(pil_img)
