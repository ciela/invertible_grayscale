import click
import torch
import torchvision.transforms as transforms

from igray.model import InvertibleGrayscale
from igray.dataset import pil_loader, DEFAULT_TRANSFORM
import igray.util as util


@click.command()
@click.option('-i', '--img_path', type=str, required=True)
@click.option('-m', '--model_path', type=str, required=True)
@click.option('-c', '--cuda_no', type=int, default=-1)
def main(img_path, model_path, cuda_no):
    model = util.load_checkpoint(model_path, cuda_no=cuda_no)
    state_dict = model['state_dict']

    ig_net = InvertibleGrayscale()
    ig_net.load_state_dict(state_dict)

    X = DEFAULT_TRANSFORM(pil_loader(img_path))
    Y_gray, Y_restored = ig_net(X.unsqueeze(0))
    Y_gray, Y_restored = (Y_gray + 1) / 2, (Y_restored + 1) / 2
    Y_gray, Y_restored = transforms.ToPILImage()(Y_gray.squeeze(0)),\
        transforms.ToPILImage(mode='RGB')(Y_restored.squeeze(0))
    Y_gray.save('gray.png')
    Y_restored.save('restored.png')


if __name__ == "__main__":
    main()
