# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
import os
from typing import Dict
import time
import datetime
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets #V
from torchvision.utils import save_image
from torch.autograd import Variable #V

from Diffusion_condition import GaussianDiffusionSampler_cond
from Model_condition import UNet
from datasets import * #V

dataset_name="brain"
out_name="trial_1"
batch_size=1
T = 1000
ch = 128
ch_mult = [1, 2, 3, 4]
attn = [2]
num_res_blocks = 2
dropout = 0.3
beta_1 = 1e-4
beta_T = 0.02
grad_clip = 1
save_weight_dir = "./Checkpoints/%s"%out_name
Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

# Create sample directories
os.makedirs("test/%s" % out_name, exist_ok=True)

# Test data loader (uses .nii.gz slices)
test_dataloader = DataLoader(
    #ImageDatasetNii(root="./brain_nii", split="train"),
    ImageDatasetNii_25D(root="./brain", split="train", k=1),
    batch_size=1,
    shuffle=False,
    num_workers=0,
)
        
def test():
    img_save = torch.Tensor()
    sampler = GaussianDiffusionSampler_cond(net_model,beta_1,beta_T,T).to(device)
    with torch.no_grad():
        for ii, batch1 in enumerate(test_dataloader):
            ii = ii + 1
            print(ii)
            #ct = Variable(batch1["pCT"].type(Tensor))
            #cbct = Variable(batch1["CBCT"].type(Tensor)) #condition 

            # Stack ct & cbct
            ct_stack = Variable(batch1["CT"].type(Tensor))         # [1,3,H,W]
            cbct_stack = Variable(batch1["CBCT"].type(Tensor))     # [1,3,H,W]
   
            #noisyImage = torch.randn(size=[1, 1, 256, 256], device=device)
            #x_in = torch.cat((noisyImage,cbct),1)

            # ----------- For 2.5D -----------
            # 1) noise ONLY the center CT slice
            noisy_center = torch.randn_like(ct_stack[:,1:2,:,:])   # [1,1,H,W]

            # 2) build the 5-channel condition
            cond = torch.cat([
                ct_stack[:,0:1,:,:],     # CT z-1
                ct_stack[:,2:3,:,:],     # CT z+1
                cbct_stack               # CBCT stack (3 slices)
            ], dim=1)                    # total 5 channels

            # 3) full 6-channel input
            x_in = torch.cat((noisy_center, cond), dim=1)

            x_out = sampler(x_in)
            
            # “inverse the normalized image to HU value”
            
            #fake = x_out[:,0,:,:]
            #fake = torch.unsqueeze(fake,1)

            #img = torch.cat((cbct,fake,ct),3)

            # -------- 2.5D --------
            fake = x_out[:,0,:,:].unsqueeze(1)                  # predicted CT(z)
            img  = torch.cat((
                cbct_stack[:,1:2,:,:],   # CBCT center slice
                fake,                    # predicted CT center
                ct_stack[:,1:2,:,:]      # GT CT center slice
            ), dim=3)

            img = img.cpu()
            img_save = torch.cat((img_save,img),1)
            img_tst = img_save.numpy()
            img_tst = img_tst.astype(np.uint16)
            img_tst.tofile("test/%s/%s.raw" % (out_name, "1000"))

net_model = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout).to(device)
#ckpt = torch.load(os.path.join(save_weight_dir, "ckpt_1000_.pt"), map_location=device)
ckpt = torch.load(os.path.join(save_weight_dir, "model_final.pt"), map_location=device)
net_model.load_state_dict(ckpt)
print("model load weight done.")
net_model.eval()
sampler = GaussianDiffusionSampler_cond(net_model,beta_1,beta_T,T).to(device)
test()