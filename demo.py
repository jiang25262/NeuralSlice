import os
from pytorch3d.io import save_obj
import random
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import join_meshes_as_batch
from model.localfea import Local_Fea_Deform, loNet, load_partial_pretrained
from loss import sli

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(198302942)


chairlist = ['./data/repre_data/chair1/points.npy','./data/repre_data/chair2/points.npy']

lrat = 1e-4
epoch = 2000


shapelist = []

for i in chairlist:

    X = np.load(i, allow_pickle=True)
    X /= np.max(np.linalg.norm(X, axis=1))   
    mask = np.random.randint(0,X.shape[0],2562) 
    X = torch.from_numpy(X[mask,...]).float()
    X = X.unsqueeze(0).to(device).float()
    shapelist.append(X)
X = torch.cat((shapelist),dim=0)


model =  Local_Fea_Deform().to(device)
modello = loNet(256).to(device)
optimizer = optim.Adam(model.parameters(), lr = lrat)
optimizer2 = optim.Adam(modello.parameters(), lr = lrat)
torch.cuda.set_device(0) 
load_partial_pretrained(model, './train_models/pretrained')

minn = 22

for e in range(epoch):

    if epoch==200:
        optimizer = optim.Adam(list(model.parameters()), lr = lrat/1.5)
    if epoch==350:
        optimizer = optim.Adam(list(model.parameters()), lr = lrat/2)
    if epoch==550:
        optimizer = optim.Adam(list(model.parameters()), lr = lrat/4)
    if epoch==1000:
        optimizer = optim.Adam(list(model.parameters()), lr = lrat/10)
    if epoch==1250:
        optimizer = optim.Adam(list(model.parameters()), lr = lrat/50)
    if epoch==1450:
        optimizer = optim.Adam(list(model.parameters()), lr = lrat/100)
    
    optimizer.zero_grad()
    optimizer2.zero_grad()
    ww = modello(X)*0.5

    pred, tetra, _ = model(X)
    loss = 0
    for i in range(X.shape[0]):

        pointlist, facelist = sli(pred[0:1,:,:], tetra[0:1,:,:],ww[i])
        mesh1 = Meshes(verts=pointlist[0],faces=facelist[0])
        pts1 = sample_points_from_meshes(meshes=mesh1,num_samples=2562)

        l1,_ = chamfer_distance(X[i:i+1,:,:],pts1)

        loss = l1 +loss
    loss/=X.shape[0]
    loss = loss.requires_grad_()

    loss.backward()
    optimizer.step()
    optimizer2.step()
    print('Epoch:',e+1,' CD:',loss.item()/2)

    if e==1999:
        i=0

        for w in torch.linspace(-0.5,0.5,50):

            pointlist, facelist = sli(pred[0:1,:,:], tetra[0:1,:,:],w)
            mesh1 = Meshes(verts=pointlist[0],faces=facelist[0])
            save_obj('./outmodel/'+str(i)+'.obj',pointlist[0][0],facelist[0][0])
            i+=1
