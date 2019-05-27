import datetime as dt

import click
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.data as data

from dataset import Dataset
from models import InvertibleGrayscale
import util
from loss import Loss


@click.command()
@click.option('-d', '--datadir', type=str)
@click.option('-c', '--cuda_no', type=int, default=-1)
def main(datadir, cuda_no):
    datestr = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d%H%M%S')
    print(f'Starting training of InvertibleGrayscale, {datestr}')

    # ready for training
    ig_net = InvertibleGrayscale()
    optim = Adam(ig_net.parameters(), lr=0.0002)  # to lr 0.000002
    scheduler = LambdaLR(optim, lr_lambda=lambda ep: 0.9623506263980885 ** ep)  # 0.01 ** (1/120)
    criterion = Loss()
    use_gpu = torch.cuda.is_available() and cuda_no >= 0
    if use_gpu:
        print(f'Use GPU No.{cuda_no}')
        gpu = torch.device('cuda', cuda_no)
        ig_net = ig_net.to(gpu)
        criterion = criterion.to(gpu)

    # start training
    data_loader = data.DataLoader(Dataset(datadir))
    for ep in range(120):
        losses = util.AverageMeter()
        for p, X_color, T_gray in data_loader:
            if use_gpu:
                X_color, T_gray = X_color.to(gpu), T_gray.to(gpu)
            optim.zero_grad()
            Y_grayscale, Y_restored = ig_net(X_color)
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
        gray.save(f'gray_ep{ep:03}.png')
        color.save(f'color_ep{ep:03}.png')
        print('Saving trained model...')
        state = {
            'epoch': ep + 1,
            'use_gpu': use_gpu,
            'state_dict': ig_net.state_dict(),
            'optimizer': optim.state_dict(),
        }
        util.save_checkpoint(state, False, datestr)
    print('Finished training!')


if __name__ == "__main__":
    main()
