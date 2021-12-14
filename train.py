import __init__
from config import *
from show_cam import *
from transform import *
from utils import *
from net import *
from dataset import *
from librosa.display import specshow
from optimizer import MADGRAD
from warmup_scheduler import GradualWarmupSchedulerV2
import pandas as pd
import numpy as np
import os
import gc
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import plotly
import plotly.express as px
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')


class Train_process(object):

    def __init__(self, config=modelConfig):
        super(Train_process, self).__init__()
        self.config = config

    def split_(self, df):
        kf = StratifiedKFold(n_splits=self.config.fold_num,
                             shuffle=True,
                             random_state=self.config.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df, df['hum_quant'])):
            df.loc[val_idx, f'fold'] = fold
        return df

    def process_data(self, df, ds_config, fold_idx):
        train_data = df[df[f'fold' ] != fold_idx].reset_index(drop=True)
        val_data = df[df[f'fold' ] == fold_idx].reset_index(drop=True)

        ex1 = np.random.choice(val_data[val_data['hum_quant'] == val_data['hum_quant'].min()].index)
        ex1 = val_data.loc[ex1, ['song_path', 'hum_path']].values
        ex3 = np.random.choice(val_data[val_data['hum_quant'] == val_data['hum_quant'].max()].index)
        ex3 = val_data.loc[ex3, ['song_path', 'hum_path']].values

        list_audio = [ex1, ex3]

        valid_dataset = humDataset(val_data, ds_config, has_transform=True, sub='validation')
        valid_loader = DataLoader(valid_dataset, pin_memory=True,
                                  batch_size = self.config.batch_size,
                                  num_workers=self.config.num_workers)

        del val_data, valid_dataset
        gc.collect()

        return train_data, valid_loader, list_audio


    def show_cam(self, model, list_audio):
        if 'efficientnet' in self.config.model_use:
            grad_cam = GradCAM(model=model.model, target_layers=[model.model.conv_head],
                               use_cuda=torch.cuda.is_available())
        elif 'resnet' in self.config.model_use:
            grad_cam = GradCAM(model=model.model, target_layers=[model.model.layer4],
                               use_cuda=torch.cuda.is_available())

        for idx, (song, hum) in enumerate(list_audio):

            # song
            song = librosa.load(song, sr=self.config.sr)[0]
            song = audioProcess(song, self.config.ds_config, alg=self.config.ds_config.alg)
            song_cam = grad_cam(song.unsqueeze(0))[0]
            song_cam = show_cam_on_image(song.detach().cpu().numpy() \
                                         .transpose(1, 2, 0 ) /255,
                                         song_cam)

            # hum

            hum = librosa.load(hum, sr=self.config.sr)[0]
            hum = audioProcess(hum, self.config.ds_config, alg=self.config.ds_config.alg)
            hum_cam = grad_cam(hum.unsqueeze(0))[0]
            hum_cam = show_cam_on_image(hum.detach().cpu().numpy() \
                                        .transpose(1, 2, 0 ) /255,
                                        hum_cam)

            fig, ax = plt.subplots(4, 1, figsize=(20, 20))
            specshow(song[0].detach().cpu().numpy(), ax=ax[0], x_axis='time', y_axis='log')
            ax[0].set_title('song')
            specshow(hum[0].detach().cpu().numpy(), ax=ax[1], x_axis='time', y_axis='log')
            ax[1].set_title('hum')

            specshow(song_cam.mean(-1), ax=ax[2], x_axis='time', y_axis='log')
            ax[2].set_title('song cam')
            specshow(hum_cam.mean(-1), ax=ax[3], x_axis='time', y_axis='log')
            ax[3].set_title('hum cam')

            plt.tight_layout()
            plt.show()


    def fit(self, df):
        os.makedirs('./save', exist_ok=True)
        fold_split_df = self.split_(df)
        fold = random.choice(np.arange(self.config.fold_num))
        print(50 *'-')
        print(f'Fold_{fold}:')
        model = humNet(self.config.model_use, emb_size=self.config.emb_size)
        model = model.to(self.config.device)
        loss_fn = nn.TripletMarginLoss(margin=0.1)
        optimizer = MADGRAD(model.parameters(), lr=self.config.lr)
        scheduler = self.config.SchedulerCosine(optimizer, **self.config.cosine_params)
        if self.config.has_warmup:
            scheduler = GradualWarmupSchedulerV2(optimizer, after_scheduler=scheduler,
                                                 **self.config.warmup_params)

        train_data, valid_loader, list_audio = \
            self.process_data(fold_split_df, self.config.ds_config, fold)

        best_val_loss = np.Inf
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark= True
        torch.cuda.empty_cache()

        list_train_loss, list_val_loss = [], []
        for epoch in range(self.config.n_epochs):
            train_dataset = humDataset(train_data, self.config.ds_config,
                                       has_transform=True, sub='train')
            train_loader = DataLoader(train_dataset,
                                      batch_size = self.config.batch_size,
                                      shuffle=True, pin_memory=True,
                                      num_workers=self.config.num_workers)

            train_loss = train_lf(self.config, train_loader, model,
                                  loss_fn, optimizer)
            val_loss = val_lf(self.config, valid_loader, model, loss_fn)
            print(f'Epoch{epoch}: Train_loss: {train_loss:.7f} | Val_loss: {val_loss:.7f}')
            list_train_loss.append(train_loss)
            list_val_loss.append(val_loss)

            if best_val_loss > val_loss:
                best_val_loss = val_loss

                torch.save(model.state_dict(), f'{args.save_path}/hum_model.pth')
                print('Model improved, saving model!')
                self.show_cam(model, list_audio)
            else:
                print('Model not improved!')

            if self.config.validation_scheduler:
                scheduler.step()

        fig = go.Figure()
        loss_df = pd.DataFrame({'epoch': np.arange(len(list_train_loss)),
                                'train_loss': list_train_loss,
                                'validation_loss': list_val_loss})
        loss_df['is_min'] = loss_df['validation_loss'] \
            .apply(lambda x :1 if x== loss_df['validation_loss'].min() else 0)

        train_data = go.Scatter(x=loss_df.epoch, y=loss_df.train_loss,
                                mode='lines+markers', name='train loss')
        val_data = go.Scatter(x=loss_df.epoch, y=loss_df.validation_loss,
                              mode='lines+markers', name='validation loss')
        layout = go.Layout(
            title="Train process",
            width=500,
            height=500)

        fig = go.Figure(data=[train_data, val_data], layout=layout)
        fig.add_annotation(x=loss_df[loss_df.is_min == 1]['epoch'].max(),
                           y=loss_df.validation_loss.min(),
                           text=f'{loss_df[loss_df.is_min == 1].epoch.max():.0f}: {loss_df.validation_loss.min():.4f}',
                           showarrow=True, arrowhead=1)

        fig.show()

        torch.cuda.empty_cache()


if __name__ == '__main__':
    os.makedirs(args.save_path, exist_ok=True)
    train_df = pd.read_csv(args.train_csv_path)
    train_df['hum_quant'] = pd.cut(train_df['hum_duration'], 5, labels=[1, 2, 3, 4, 5])
    train_df['list_negative'] = train_df \
        .apply(lambda x: [i for i in train_df[train_df.hum_quant == x.hum_quant].song_path \
                          if i != x.song_path], axis=1)
    print('Loading train file done!')
    train_pr = Train_process()
    train_pr.fit(train_df)