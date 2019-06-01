from typing import Tuple
import shutil

import PIL.Image
import torch
import torchvision.transforms as transforms


def tensor_to_img(orig: torch.Tensor, gray: torch.Tensor, restored: torch.Tensor) -> Tuple:
    orig, gray, restored = (orig + 1) / 2, (gray + 1) / 2, (restored + 1) / 2
    orig = transforms.ToPILImage(mode='RGB')(orig)
    gray = transforms.ToPILImage()(gray)
    restored = transforms.ToPILImage(mode='RGB')(restored)
    return orig, gray, restored


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


def save_checkpoint(state: dict, is_best: bool, datestr: str):
    """Save checkpoint state data into file.
    Arguments:
        state {dict} -- [description]
        is_best {bool} -- [description]
        datestr {str} -- [description]
    """
    filename = 'igray_checkpoint_' + datestr + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'iqcrnet_best_' +
                        datestr + '.pth.tar')


def load_checkpoint(filepath: str, cuda_no: int = -1) -> dict:
    use_gpu = torch.cuda.is_available() and cuda_no >= 0
    if use_gpu:
        state = torch.load(filepath, map_location=f'cuda:{cuda_no}')
    else:
        state = torch.load(filepath, map_location='cpu')
    return state
