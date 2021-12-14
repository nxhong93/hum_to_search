import os

from config import *
from transform import *
from utils import *
from net import *
from dataset import *
import librosa
from collections import OrderedDict
import sys


def inference():
    list_audio = []
    test_emb = []

    model = humNet('tf_efficientnet_b0_ns', emb_size=modelConfig.emb_size,
                   model_weight=args.model_weight0, is_train=False)
    model = model.to(modelConfig.device)
    model.eval()

    duration = modelConfig.ds_config.max_time * modelConfig.sr
    test_song_path = os.path.join(args.test_path, 'full_song')
    test_hum_path = os.path.join(args.test_path, 'hum')

    for song_path in tqdm(glob(f'{test_song_path}/*.mp3'), leave=False):
        song = song_path.split('/')[-1][:-4]
        data = librosa.load(song_path, sr=SR)[0]
        num_frames = 1 + data.shape[0] // duration
        for idx, num_frame in enumerate(range(num_frames)):
            list_audio.append(song)

            if idx == num_frames - 1:
                frame = data[-duration:]
            else:
                frame = data[idx * duration:(idx + 1) * duration]
            frame = audioProcess(frame, modelConfig.ds_config, alg=modelConfig.ds_config.alg)
            frame = frame.unsqueeze(0).to(modelConfig.device)
            with torch.no_grad():
                frame = model(frame).squeeze(0).detach().cpu().numpy()
            test_emb.append(frame)

    test_emb = np.array(test_emb)

    test_emb1 = []
    model1 = humNet('tf_efficientnet_b1_ns', emb_size=modelConfig.emb_size,
                    model_weight=args.model_weight1, is_train=False)
    model1 = model1.to(modelConfig.device)
    model1.eval()

    duration = modelConfig.ds_config.max_time * modelConfig.sr
    for song_path in tqdm(glob(f'{test_song_path}/*.mp3'), leave=False):
        song = song_path.split('/')[-1][:-4]
        data = librosa.load(song_path, sr=SR)[0]
        num_frames = 1 + data.shape[0] // duration
        for idx, num_frame in enumerate(range(num_frames)):

            if idx == num_frames - 1:
                frame = data[-duration:]
            else:
                frame = data[idx * duration:(idx + 1) * duration]
            frame = audioProcess(frame, modelConfig.ds_config, alg=modelConfig.ds_config.alg)
            frame = frame.unsqueeze(0).to(modelConfig.device)
            with torch.no_grad():
                frame = model1(frame).squeeze(0).detach().cpu().numpy()
            test_emb1.append(frame)

    test_emb1 = np.array(test_emb1)

    test_all = np.concatenate([test_emb, test_emb1], 1)
    list_hum = []
    list_hum_to_song = []

    for hum_path in tqdm(sorted(glob(f'{test_hum_path}/*.mp3')), leave=False):
        hum_id = hum_path.split('/')[-1]
        list_hum.append(hum_id)
        hum = librosa.load(hum_path, sr=SR)[0]

        number_hums = 1 + hum.shape[0] // duration
        distance_emb = []
        for idx, num_hum in enumerate(range(number_hums)):
            if idx == number_hums - 1:
                frame = hum[-duration:]
            else:
                frame = hum[idx * duration:(idx + 1) * duration]
            frame = audioProcess(frame, modelConfig.ds_config, alg=modelConfig.ds_config.alg)
            frame = frame.unsqueeze(0).to(modelConfig.device)
            with torch.no_grad():
                frame0 = model(frame).squeeze(0).detach().cpu().numpy()
                frame1 = model1(frame).squeeze(0).detach().cpu().numpy()
                frame = np.concatenate([frame0, frame1])
            distance = np.linalg.norm(test_all - frame, axis=1)
            distance_emb.append(distance)
        distance_emb = np.array(distance_emb)
        song_hum = list_audio[distance_emb.argmin() % len(list_audio)]
        list_hum_to_song.append(song_hum)


    os.makedirs('./result/', exist_ok=True)
    sub_df = pd.DataFrame({'hum': list_hum, 'song': list_hum_to_song})
    sub_df.to_csv('./result/submission.csv', header=False, index=False)

    sub_df.head()

if __name__ == '__main__':
    inference()