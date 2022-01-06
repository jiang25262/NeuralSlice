import torch
from src.encoder.pointnet import LocalPoolPointnet
import torch.nn as nn
import torch.nn.functional as F
from src.decoder import LocalDecoder
import numpy as np
class loNet(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super(loNet, self).__init__()
        
        self.zdim = zdim
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, zdim, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)

        self.fc_bn = nn.BatchNorm1d(zdim)
        self.fc_bn2 = nn.BatchNorm1d(zdim//2)

        self.fc = nn.Linear(zdim, zdim)
        self.fc2 = nn.Linear(zdim, zdim//2)
        self.fc3 = nn.Linear(zdim//2, 1)

    def forward(self, x):

        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.zdim)
        
        ms = F.relu(self.fc_bn(self.fc(x)))
        ms = F.relu(self.fc_bn2(self.fc2(ms)))
        ms = torch.tanh(self.fc3(ms))
        return ms


class Local_Fea_Deform(nn.Module):

    def __init__(self, cdim = 32, hiddendim = 32):
        super(Local_Fea_Deform, self).__init__()

        print("Local_Fea_Deform embedding initialized")

        self.encoder = LocalPoolPointnet(c_dim=cdim, dim=3, hidden_dim=hiddendim, scatter_type='max', 
                        unet3d=True, unet3d_kwargs={'num_levels': 3,'f_maps': 32,'in_channels': 32,'out_channels': 32}, 
                        grid_resolution=32, plane_type='grid', padding=0.1, n_blocks=5).float()

        self.decoder = LocalDecoder(dim=4, c_dim=cdim,
                        hidden_size=hiddendim, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1).float()

    def forward(self, p):

        grid = self.encoder(p)
        out, face, c = self.decoder(p,grid)

        return out, face, c


def load_partial_pretrained(mymodel, path):

    pretrained_dict = torch.load(path)
    model_dict = mymodel.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict) 
    mymodel.load_state_dict(model_dict)
    print('Load pretrained model: ',path)
