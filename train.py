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
@click.option('-m', '--max_epoch', type=int, default=120)
@click.option('-f', '--chkpt_file', type=str, default=None)
def main(datadir, cuda_no, num_samples, max_epoch, chkpt_file):
    datestr = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d%H%M%S')
    logzero.logfile(f'igray_train_{datestr}.log', maxBytes=10e6, backupCount=3)
    log.info(f'Starting training of InvertibleGrayscale, {datestr}')
    use_gpu = torch.cuda.is_available() and cuda_no >= 0
    if use_gpu:
        log.info(f'Use GPU No.{cuda_no}')
    writer = SummaryWriter()
    state = None
    if chkpt_file:
        log.info(f'Resuming training from checkpoint file {chkpt_file}')
        state = util.load_checkpoint(chkpt_file, cuda_no=cuda_no)

    # ready for training
    ig_net = InvertibleGrayscale()
    if state:
        ig_net.load_state_dict(state['state_dict'])
    if use_gpu:
        gpu = torch.device('cuda', cuda_no)
        ig_net = ig_net.to(gpu)
    optim = Adam(ig_net.parameters(), lr=0.0002)  # to lr 0.000002
    scheduler = None
    if state:
        optim.load_state_dict(state['optimizer'])
    else:
        scheduler = LambdaLR(optim, lr_lambda=lambda ep: (0.01 ** (1/max_epoch)) ** ep)
    criterion = Loss()
    if use_gpu:
        criterion = criterion.to(gpu)

    # start training
    data_loader = data.DataLoader(
        Dataset(datadir, num_samples=num_samples), shuffle=True, pin_memory=True)
    img_to_track = None
    for ep in range(state['epoch'], max_epoch) if state else range(max_epoch):
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
            # track learning progress
            if img_to_track is None:
                img_to_track = p[0]
                log.info(f'Set images to track to {img_to_track}')
                track_progress(X_color, Y_grayscale, Y_restored, writer, ep)
            elif p[0] == img_to_track:
                # save current information
                log.info(f'Log generated images of {img_to_track}')
                track_progress(X_color, Y_grayscale, Y_restored, writer, ep)
        if scheduler:
            scheduler.step()
        log.info(f'EP{ep:03}STG{1 if ep < 90 else 2}: LossAvg: {losses.avg}')
        writer.add_scalar('igray/train_loss', losses.avg, ep)
        log.info('Saving trained model')
        state = {
            'epoch': ep + 1,
            'use_gpu': use_gpu,
            'state_dict': ig_net.state_dict(),
            'optimizer': optim.state_dict(),
        }
        util.save_checkpoint(state, False, datestr)
    log.info('Finished training!')


def track_progress(orig, gray, restored, writer, ep):
    orig, gray, restored = orig.squeeze(0), gray.squeeze(0), restored.squeeze(0)
    writer.add_image('igray/img_orig', (orig + 1) / 2, ep)
    writer.add_image('igray/img_gray', (gray + 1) / 2, ep)
    writer.add_image('igray/img_restored', (restored + 1) / 2, ep)


if __name__ == "__main__":
    main()
