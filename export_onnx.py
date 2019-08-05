import click
import torch
import torch.onnx

from igray.model import InvertibleGrayscale
import igray.util as util


DUMMY = torch.randn(1, 1, 256, 256)


@click.command()
@click.option('-m', '--model_path', type=str, required=True)
@click.option('-c', '--cuda_no', type=int, default=-1)
def main(model_path: str, cuda_no: int):
    model = util.load_checkpoint(model_path, cuda_no=cuda_no)
    state_dict = model['state_dict']

    ig_net = InvertibleGrayscale()
    ig_net.load_state_dict(state_dict)

    dummy = torch.randn(1, 1, 256, 256)
    torch.onnx.export(
        ig_net.decoder, DUMMY, 'igray_decoder.onnx', verbose=True)


if __name__ == "__main__":
    main()
