import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import nnAudio
from nnAudio.Spectrogram import MFCC, STFT, MelSpectrogram, CQT, CQT2010v2
from tqdm import tqdm


def audioProcess(data, config, alg='mel'):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    if alg == 'mel':
        S = MelSpectrogram(**config.melspectrogram_config)(data)
        wave = librosa.power_to_db(S, ref=np.max)
        wave = mono_to_color(wave) / 255
    elif alg == 'mfcc':
        wave = MFCC(**config.mfcc_config)(data)
    elif alg == 'cqt':
        wave = CQT2010v2(**config.cqt_config)(data)
    wave = wave.clone().detach().float()
    return wave


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_lf(cfg, train_loader, model,
             loss_fn, optimizer):
    model.train()
    if cfg.apex:
        scaler = GradScaler()

    summary_loss = AverageMeter()
    for idx, (hum, pos, neg) in tqdm(enumerate(train_loader),
                                     total=len(train_loader),
                                     leave=False):
        pos = pos.to(cfg.device)
        neg = neg.to(cfg.device)
        hum = hum.to(cfg.device)
        optimizer.zero_grad()
        if cfg.apex:
            with autocast():
                out_pos = model(pos)
                out_neg = model(neg)
                out_hum = model(hum)
                loss = loss_fn(out_hum, out_pos, out_neg)
            scaler.scale(loss).backward()
        else:
            out_pos = model(pos)
            out_neg = model(neg)
            out_hum = model(hum)
            loss = loss_fn(out_hum, out_pos, out_neg)
            loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm)
        summary_loss.update(loss.detach().item(), hum.shape[0])

        if idx % cfg.accumulation_step == 0:
            if cfg.apex:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

    return summary_loss.avg


def val_lf(cfg, val_loader, model, loss_fn):
    model.eval()
    summary_loss = AverageMeter()
    for idx, (hum, pos, neg) in tqdm(enumerate(val_loader),
                                     total=len(val_loader),
                                     leave=False):
        pos = pos.to(cfg.device)
        neg = neg.to(cfg.device)
        hum = hum.to(cfg.device)

        with torch.no_grad():
            out_pos = model(pos)
            out_neg = model(neg)
            out_hum = model(hum)
            loss = loss_fn(out_hum, out_pos, out_neg)

        summary_loss.update(loss.detach().item(), hum.shape[0])

    return summary_loss.avg
