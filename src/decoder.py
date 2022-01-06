import torch
from src.encoder.pointnet import LocalPoolPointnet
from src.layers import ResnetBlockFC
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate, normalize_coord,normalize_coordinate


class PointNet(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super(PointNet, self).__init__()
        
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, zdim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(zdim)

        self.fc1 = nn.Linear(zdim, zdim)
        self.fc2 = nn.Linear(zdim, zdim)
        self.fc_bn1 = nn.BatchNorm1d(zdim)

    def forward(self, x):

        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)),inplace=True)
        x = F.relu(self.bn2(self.conv2(x)),inplace=True)
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.zdim)
        
        ms = F.relu(self.fc_bn1(self.fc1(x)),inplace=True)
        ms = self.fc2(ms)

        return ms

class Map(nn.Module):
    def __init__(self, zdim, point_dim=4):
        super(Map, self).__init__()

        self.conv1 = nn.Conv1d((point_dim+zdim), (point_dim+zdim)//2, 1)
        self.conv2 = nn.Conv1d((point_dim+zdim)//2, (point_dim+zdim)//4, 1)
        self.conv3 = nn.Conv1d((point_dim+zdim)//4, (point_dim+zdim)//8, 1)
        self.conv4 = nn.Conv1d((point_dim+zdim)//8, 3, 1)

        self.bn1 = nn.BatchNorm1d((point_dim+zdim)//2)
        self.bn2 = nn.BatchNorm1d((point_dim+zdim)//4)
        self.bn3 = nn.BatchNorm1d((point_dim+zdim)//8)

    def forward(self, input, latent):

        x = input.transpose(1,2)
        latent = latent.repeat(1,1,x.size()[2])
        x = torch.cat((latent,x),dim=1)
        x = F.relu(self.bn1(self.conv1(x)),inplace=True)
        x = F.relu(self.bn2(self.conv2(x)),inplace=True) 
        x = F.relu(self.bn3(self.conv3(x)),inplace=True) 
        x = torch.tanh(self.conv4(x))
        x = x.transpose(1,2)

        return x

class Locating(nn.Module):

    def __init__(self, zdim=256):
        super(Locating, self).__init__()

        self.v, self.t = read_4obj('./model/tour_small.4obj')
        self.t = torch.from_numpy(self.t).long()
        self.v = torch.from_numpy(self.v) / 6
        self.encoder = PointNet(zdim).float()  
        self.decoder = Map(zdim=zdim).float()

    def forward(self, input):

        batch_size = input.shape[0]
        latent = self.encoder(input)
        vertices = self.v.unsqueeze(0).repeat(batch_size,1,1).type_as(input)
        face = self.t.unsqueeze(0).repeat(batch_size,1,1).type_as(input)
        outs = self.decoder(vertices,latent.unsqueeze(2))

        return outs, vertices, face, latent.unsqueeze(2)

class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=8, c_dim=128,
                 hidden_size=256, n_blocks=4, leaky=False, sample_mode='bilinear', padding=0.1, zdim = 1000):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        self.dim = dim

        self.locate = Locating(256)

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.BatchNorm = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size,dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_plane):

        Lo, vertices, face, latent = self.locate(p)

        c = self.sample_grid_feature(Lo, c_plane['grid'])
        c = c.transpose(1, 2)


        net = self.fc_p(vertices)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = torch.tanh(self.fc_out(self.actvn(net)))


        return out[:,:,0:4], face, c



def read_4obj(filename):

    fin = open(str(filename))
    lines = fin.readlines()
    fin.close()

    vertices = []
    tetrahedrons = []

    for i in range(len(lines)):
        line = lines[i].split()
        if len(line)==0:
            continue
        if line[0] == 'v':
            x = float(line[1])
            y = float(line[2])
            z = float(line[3])
            w = float(line[4])
            vertices.append([x,y,z,w])
        if line[0] == 't':
            x = int(line[1].split("/")[0])
            y = int(line[2].split("/")[0])
            z = int(line[3].split("/")[0])
            w = int(line[4].split("/")[0])
            tetrahedrons.append([x,y,z,w])

    vertices = np.array(vertices, np.float32)
    tetrahedrons = np.array(tetrahedrons, np.float32)

    return vertices, tetrahedrons