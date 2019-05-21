import click
import torch

from models import Encoder, Decoder
import util
from loss import Loss


@click.command()
@click.option('-i', '--imgpath', type=str)
def main(imgpath):
    gpu = torch.device('cuda:0')
    criterion = Loss(weights=(1, 1, 1), gc_weights=(1, 1, 1))
    encoder = Encoder().to(gpu)
    print(list(encoder.parameters()))
    img_tensor = util.img_to_tensor(img_path=imgpath).unsqueeze(0).to(gpu)
    grayscale = encoder(img_tensor)
    print(grayscale.size())
    decoder = Decoder().to(gpu)
    print(list(decoder.parameters()))
    resotred = decoder(grayscale)
    print(resotred.size())
    loss = criterion(img_tensor, grayscale, resotred)
    print(loss)
    loss.backward()


if __name__ == "__main__":
    main()
