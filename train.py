import click
import torch

from models import InvertibleGrayscale
import util
from loss import Loss


@click.command()
@click.option('-i', '--imgpath', type=str)
def main(imgpath):
    gpu = torch.device('cuda:0')
    criterion = Loss(weights=(1, 1, 1), gc_weights=(1, 1, 1))
    invertible_grayscale = InvertibleGrayscale().to(gpu)
    print(invertible_grayscale)
    img_tensor = util.img_to_tensor(img_path=imgpath).unsqueeze(0).to(gpu)
    grayscale, restored = invertible_grayscale(img_tensor)
    print(grayscale.size(), restored.size())
    loss = criterion(img_tensor, grayscale, restored)
    print(loss)
    loss.backward()


if __name__ == "__main__":
    main()
