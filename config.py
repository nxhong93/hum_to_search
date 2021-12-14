import argparse
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('--train_path', default='./input/train/')
parser.add_argument('--test_path', default='./input/private_test/')
parser.add_argument('--sub_path', default='./input/sample_predict/')

parser.add_argument('--train_hum_path', default='./input/train/hum')
parser.add_argument('--train_song_path', default='./input/train/song')
parser.add_argument('--train_csv_path', default='./input/train/train_meta.csv')

parser.add_argument('--save_path', default='./src/saved_models')
parser.add_argument('--model_weight0', default='./src/saved_models/hum_model_cqt.pth')
parser.add_argument('--model_weight1', default='./src/saved_models/hum_model_cqt1.pth')
args = parser.parse_args()


SR = 44100

class DatasetConfig:
    sr = SR
    alg = 'cqt'
    max_time = 10
    melspectrogram_config = {'sr': SR,
                             'verbose': False,
                             'hop_length': 512,
                             'n_fft': 2048,
                             'n_mels': 128,
                             'fmin': 20,
                             'fmax': 20000}
    mfcc_config = {'sr': SR,
                   'verbose': False,
                   'n_mfcc': 39}
    cqt_config = {'sr': SR,
                  'hop_length': 512,
                  'bins_per_octave': 12,
                  'verbose': False,
                  'fmin': 20,
                  'fmax': None}


class modelConfig:
    sr = SR
    model_use = 'tf_efficientnet_b0_ns'
    emb_size = 1024
    fold_num = 5
    ds_config = DatasetConfig
    batch_size = 6
    predict_batch_size = 2*batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 8
    seed = 666
    lr = 1e-4
    n_epochs = 200
    verbose = 1
    verbose_step = 1
    grad_max_norm = 3
    accumulation_step = 1
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.2,
        patience=1,
        threshold_mode='abs',
        min_lr=1e-7
    )
    SchedulerCosine = CosineAnnealingLR
    cosine_params = dict(
        T_max=n_epochs,
        eta_min=1e-7,
        verbose=True
    )
    has_warmup = True
    warmup_params = {
        'multiplier': 10,
        'total_epoch': 1,
    }
    grad_cam_step = 3
    apex = False