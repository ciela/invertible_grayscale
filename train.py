import click
import torch

from models import InvertibleGrayscale
import util
from loss import Loss


@click.command()
@click.option('-i', '--imgpath', type=str)
def main(imgpath):
    gpu = torch.device('cuda:0')
    criterion = Loss(weights1=(1, 1, 1), weights2=(1, 1, 1), gc_weights=(1, 1, 1))
    invertible_grayscale = InvertibleGrayscale()
    pil_img = util.pil_loader(img_path=imgpath)
    X_color, T_gray = util.img_to_tensor(pil_img=pil_img)
    X_color, T_gray = X_color.unsqueeze(0), T_gray.unsqueeze(0)
    Y_grayscale, Y_restored = invertible_grayscale(X_color)
    loss = criterion(X_color, T_gray, Y_grayscale, Y_restored, stage=1)
    print(loss)
    loss.backward()


if __name__ == "__main__":
    main()
