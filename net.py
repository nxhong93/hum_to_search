import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from collections import OrderedDict
from glob import glob
import warnings

warnings.filterwarnings('ignore')


class humNet(nn.Module):
    def __init__(self, model_name, emb_size=512, model_weight=None, is_train=True):
        super(humNet, self).__init__()
        self.model = timm.create_model(model_name, pretrained=is_train, in_chans=1)
        if is_train:
            if model_weight is not None:
                state = torch.load(model_weight, map_location=lambda storage, loc: storage)
                self.model.load_state_dict(state)
            if list(self.model.named_modules())[-1][0] == 'classifier':
                self.model.classifier = nn.Linear(in_features= \
                                                      self.model.classifier.in_features,
                                                  out_features=emb_size)
            elif list(self.model.named_modules())[-1][0] == 'fc':
                self.model.fc = nn.Linear(in_features=self.model.fc.in_features,
                                          out_features=emb_size)
        else:
            if list(self.model.named_modules())[-1][0] == 'classifier':
                self.model.classifier = nn.Linear(in_features= \
                                                      self.model.classifier.in_features,
                                                  out_features=emb_size)
            elif list(self.model.named_modules())[-1][0] == 'fc':
                self.model.fc = nn.Linear(in_features=self.model.fc.in_features,
                                          out_features=emb_size)
            if model_weight is not None:
                new_keys = self.model.state_dict().keys()
                values = torch.load(model_weight,
                                    map_location=lambda storage, loc: storage).values()
                self.model.load_state_dict(OrderedDict(zip(new_keys, values)))

    def forward(self, audio):
        out = self.model(audio)
        out = F.normalize(out)
        return out