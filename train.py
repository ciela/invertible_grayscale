import click
import torch
from torch.optim.adam import Adam

from models import InvertibleGrayscale
import util
from loss import Loss


@click.command()
@click.option('-i', '--imgpath', type=str)  # TODO: Specify data directory
@click.option('-c', '--cuda_no', type=int, default=-1)
def main(imgpath, cuda_no):
    use_gpu = torch.cuda.is_available() and cuda_no >= 0
    if use_gpu:
        print(f'Use GPU No. {cuda_no}.')
        gpu = torch.device('cuda', cuda_no)
    ig = InvertibleGrayscale()
    optim = Adam(ig.parameters(), lr=0.0002)
    criterion = Loss()
    if use_gpu:
        ig = ig.to(gpu)
        criterion = criterion.to(gpu)
    for ep in range(120):
        # TODO: Create Dataloaders
        pil_img = util.pil_loader(img_path=imgpath)
        X_color, T_gray = util.img_to_tensor(pil_img=pil_img)
        X_color, T_gray = X_color.unsqueeze(0), T_gray.unsqueeze(0)
        if use_gpu:
            X_color, T_gray = X_color.to(gpu), T_gray.to(gpu)
        optim.zero_grad()
        Y_grayscale, Y_restored = ig(X_color)
        if ep < 90:
            loss = criterion(X_color, T_gray, Y_grayscale, Y_restored, stage=1)
        else:
            loss = criterion(X_color, T_gray, Y_grayscale, Y_restored, stage=2)
        print(loss)
        loss.backward()
        optim.step()
    print('Finished training!')


if __name__ == "__main__":
    main()
