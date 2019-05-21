from typing import Tuple

import PIL.Image
import torch
import torchvision.transforms as transforms


DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


GRAYSCALE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


def pil_loader(img_path: str) -> PIL.Image:
    with open(img_path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')


def img_to_tensor(pil_img: PIL.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    return DEFAULT_TRANSFORM(pil_img), GRAYSCALE_TRANSFORM(pil_img)


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
