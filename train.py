import click
import torch

from models import Encoder, Decoder
import util


@click.command()
@click.option('-i', '--imgpath', type=str)
def main(imgpath):
    encoder = Encoder()
    print(encoder)
    img_tensor = util.img_to_tensor(img_path=imgpath).unsqueeze(0)
    grayscale = encoder(img_tensor)
    print(grayscale.size())
    decoder = Decoder()
    print(decoder)
    resotred = decoder(grayscale)
    print(resotred.size())


if __name__ == "__main__":
    main()
