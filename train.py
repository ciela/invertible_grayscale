import datetime as dt

import click
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR

from models import InvertibleGrayscale
import util
from loss import Loss


@click.command()
@click.option('-i', '--imgpath', type=str)  # TODO: Specify data directory
@click.option('-c', '--cuda_no', type=int, default=-1)
def main(imgpath, cuda_no):
    datestr = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d%H%M%S')
    print(f'Starting training of InvertibleGrayscale, {datestr}')
    use_gpu = torch.cuda.is_available() and cuda_no >= 0
    if use_gpu:
        print(f'Use GPU No.{cuda_no}')
        gpu = torch.device('cuda', cuda_no)
    ig = InvertibleGrayscale()
    optim = Adam(ig.parameters(), lr=0.0002)  # to lr 0.000002
    scheduler = LambdaLR(optim, lr_lambda=lambda ep: 0.9623506263980885 ** ep)  # 0.01 ** (1/120)
    criterion = Loss()
    if use_gpu:
        ig = ig.to(gpu)
        criterion = criterion.to(gpu)
    for ep in range(120):
        # TODO: Create Dataloaders
        losses = util.AverageMeter()
        for i in range(2):
            pil_img = util.pil_loader(img_path=imgpath)
            X_color, T_gray = util.img_to_tensor(pil_img=pil_img)
            X_color, T_gray = X_color.unsqueeze(0), T_gray.unsqueeze(0)
            if use_gpu:
                X_color, T_gray = X_color.to(gpu), T_gray.to(gpu)
            optim.zero_grad()
            Y_grayscale, Y_restored = ig(X_color)
            loss = criterion(
                X_color, T_gray, Y_grayscale, Y_restored,
                stage=1 if ep < 90 else 2)
            loss.backward()
            optim.step()
            losses.update(loss)
        scheduler.step()
        print(f'EP{ep:03}STG{1 if ep < 90 else 2}: LossAvg: {losses.avg}')
        print('Saving invertible grayscale and restored color image...')
        gray, color = util.tensor_to_img(
            Y_grayscale.squeeze(0).cpu(), Y_restored.squeeze(0).cpu())
        gray.save('gray.png')
        color.save('color.png')
        print('Saving trained model...')
        state = {
            'epoch': ep + 1,
            'use_gpu': use_gpu,
            'state_dict': ig.state_dict(),
            'optimizer': optim.state_dict(),
        }
        util.save_checkpoint(state, False, datestr)
    print('Finished training!')


if __name__ == "__main__":
    main()
