import datetime as dt

import click
import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.data as data
import logzero
from logzero import logger as log
from tensorboardX import SummaryWriter

from igray.dataset import Dataset
from igray.model import InvertibleGrayscale
import igray.util as util
from igray.loss import Loss


@click.command()
@click.option('-d', '--datadir', type=str)
@click.option('-c', '--cuda_no', type=int, default=-1)
@click.option('-n', '--num_samples', type=int, default=-1)
def main(datadir, cuda_no, num_samples):
    datestr = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d%H%M%S')
    logzero.logfile(f'igray_train_{datestr}.log', maxBytes=10e6, backupCount=3)
    log.info(f'Starting training of InvertibleGrayscale, {datestr}')
    writer = SummaryWriter()

    # ready for training
    ig_net = InvertibleGrayscale()
    optim = Adam(ig_net.parameters(), lr=0.0002)  # to lr 0.000002
    scheduler = LambdaLR(optim, lr_lambda=lambda ep: 0.9623506263980885 ** ep)  # 0.01 ** (1/120)
    criterion = Loss()
    use_gpu = torch.cuda.is_available() and cuda_no >= 0
    if use_gpu:
        log.info(f'Use GPU No.{cuda_no}')
        gpu = torch.device('cuda', cuda_no)
        ig_net = ig_net.to(gpu)
        criterion = criterion.to(gpu)

    # start training
    data_loader = data.DataLoader(
        Dataset(datadir, num_samples=num_samples), shuffle=True, pin_memory=True)
    for ep in range(120):
        # calculate losses and perform backprop
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
        log.info(f'EP{ep:03}STG{1 if ep < 90 else 2}: LossAvg: {losses.avg}')
        writer.add_scalar('igray/train_loss', losses.avg, ep)

        # save current information
        log.info(f'Saving last invertible grayscale and restored color image of {p[0]}')
        if use_gpu:
            orig, gray, restored = X_color.squeeze(0).cpu(), Y_grayscale.squeeze(0).cpu(), Y_restored.squeeze(0).cpu()
        else:
            orig, gray, restored = X_color.squeeze(0), Y_grayscale.squeeze(0), Y_restored.squeeze(0)
        writer.add_image('igray/img_orig', (orig + 1) / 2, ep)
        writer.add_image('igray/img_gray', (gray + 1) / 2, ep)
        writer.add_image('igray/img_restored', (restored + 1) / 2, ep)
        orig, gray, restored = util.tensor_to_img(orig, gray, restored)
        orig.save(f'train_results/orig_ep{ep:03}.png')
        gray.save(f'train_results/gray_ep{ep:03}.png')
        restored.save(f'train_results/restored_ep{ep:03}.png')
        log.info('Saving trained model')
        state = {
            'epoch': ep + 1,
            'use_gpu': use_gpu,
            'state_dict': ig_net.state_dict(),
            'optimizer': optim.state_dict(),
        }
        util.save_checkpoint(state, False, datestr)
    log.info('Finished training!')


if __name__ == "__main__":
    main()
