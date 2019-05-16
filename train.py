import torch

from models import Encoder, Decoder


# TODO: delete
def main():
    encoder = Encoder()
    print(encoder)
    # pil_img = util.pil_loader('image_path')
    # img_tensor = util.DEFAULT_TRANSFORM(pil_img).unsqueeze(0)
    img_tensor = torch.randn((3, 256, 256)).view(1, 3, 256, 256)
    grayscale = encoder(img_tensor)
    print(grayscale)
    decoder = Decoder()
    print(decoder)
    resotred = decoder(grayscale)
    print(resotred)


if __name__ == "__main__":
    main()
