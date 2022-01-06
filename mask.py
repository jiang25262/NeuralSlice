from os import write
import numpy as np
import torch
from torch._C import device
from goto import with_goto

device = torch.device('cpu')
def write_obj(pointls,meshs = []):

    fout = open('1.obj', 'w')

    for i in range(0,len(pointls)):

        fout.write('v' + ' '+str(pointls[i,0]) + ' ' + str(pointls[i,1]) + ' ' + str(pointls[i,2])+"\n")

    for i in range(len(meshs)):

        mesh = meshs[i]
        fout.write('f' + ' ' + str(mesh[0]) + ' ' + str(mesh[1]) + ' ' + str(mesh[2])+"\n")

    fout.close()

def random_mask(reso,prob):
    all = torch.arange(0,reso**3)
    index = torch.randperm(all.size(0))
    mask = all[index]
    return mask[0:int(len(mask)*prob)]

def generate_masked_points(orx, reso,prob):
    x=orx
    x = x/2
    x += 0.5
    x = (x * reso).long()
    index = x[:,:,0] + reso * (x[:,:,1]+ reso * x[:,:,2])
    index_num = index.cpu().numpy()
    index = index[:,:,None]
    maskedlist = []
    masklist = []
    min_p = 9e20
    for i in range(x.size(0)):

        mask = np.unique(index_num[i:i+1])
        mask = torch.from_numpy(mask)
        all = torch.arange(0,mask.size(0))
        index_m = torch.randperm(all.size(0))
        mask = mask[index_m]
        mask = mask[0:int(len(mask)*prob)].view(1,1,-1)
        masklist.append(mask)
        mask = mask.repeat(1,index.size(1),1)

        a = torch.nonzero(index[i:i+1,:,:]==mask)
        maskedlist.append(orx[a[:,0],a[:,1],:])
        if a.size(0)<min_p:
            min_p=len(a)

    index = torch.arange(min_p)
    index = torch.randperm(index.size(0))

    for i in range(x.size(0)):
        maskedlist[i]=maskedlist[i][index].unsqueeze(0)

    return torch.cat(maskedlist,dim=0), masklist


chairlist = ['./data/repre_data/chair1/points.npy',
                './data/repre_data/chair2/points.npy']

lrat = 1e-4
epoch = 2000


shapelist = []

for i in chairlist:

    XX = np.load(i, allow_pickle=True)
    XX /= np.max(np.linalg.norm(XX, axis=1))   
    mask = np.random.randint(0,XX.shape[0],2562) 
    XX = torch.from_numpy(XX[mask,...]).float()
    XX = XX.unsqueeze(0).to(device).float()
    shapelist.append(XX)
  
XX = torch.cat((shapelist),dim=0)

write_obj(XX[0].numpy())
num_mask = 8
prob = 0.25
mask = random_mask(num_mask,prob)
points,_ = generate_masked_points(XX,num_mask,prob)
print(points[0])
write_obj(points[0].numpy())


