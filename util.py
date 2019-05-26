from typing import Tuple

import PIL.Image
import torch
import torchvision.transforms as transforms


ImageTensors = Tuple[torch.Tensor, torch.Tensor]


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


def pil_loader(img_path: str) -> PIL.Image:
    with open(img_path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')


def img_to_tensor(pil_img: PIL.Image) -> ImageTensors:
    return DEFAULT_TRANSFORM(pil_img), GRAYSCALE_TRANSFORM(pil_img)


def tensor_to_img(gray: torch.Tensor, restored: torch.Tensor) -> Tuple:
    gray, restored = (gray + 1) / 2, (restored + 1) / 2
    gray = transforms.ToPILImage()(gray)
    restored = transforms.ToPILImage(mode='RGB')(restored)
    return gray, restored


class AverageMeter(object):
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all measurable fields.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """Update fields by giving value and data size.
        Arguments:
            val {float} -- The value which used to compute average.
        Keyword Arguments:
            n {int} -- Size of data. (default: {1})
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
