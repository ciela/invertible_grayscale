import click
import torch
from torch.optim.adam import Adam

from models import InvertibleGrayscale
import util
from loss import Loss


@click.command()
@click.option('-i', '--imgpath', type=str)
def main(imgpath):
    gpu = torch.device('cuda:0')
    invertible_grayscale = InvertibleGrayscale()
    print(invertible_grayscale)
    optim = Adam(invertible_grayscale.parameters(), lr=0.0002)
    criterion = Loss()
    pil_img = util.pil_loader(img_path=imgpath)
    X_color, T_gray = util.img_to_tensor(pil_img=pil_img)
    X_color, T_gray = X_color.unsqueeze(0), T_gray.unsqueeze(0)
    for i in range(10):
        optim.zero_grad()
        Y_grayscale, Y_restored = invertible_grayscale(X_color)
        loss = criterion(X_color, T_gray, Y_grayscale, Y_restored, stage=1)
        print(loss)
        loss.backward()
        optim.step()


if __name__ == "__main__":
    main()
