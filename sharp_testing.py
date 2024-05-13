import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as f


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        super().__init__()
        self.en_conv1=nn.Conv2d(1,16,3,padding=1)
        self.en_conv2=nn.Conv2d(16,4,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.de_conv1=nn.ConvTranspose2d(4,16,2,stride=2)
        self.de_conv2=nn.ConvTranspose2d(16,1,2,stride=2)

    def forward(self,x):
        x=self.en_conv1(x)
        x=f.relu(x)
        x=self.pool(x)
        x=self.en_conv2(x)
        x=f.relu(x)
        x=self.pool(x)
        x=self.de_conv1(x)
        x=f.relu(x)
        x=self.de_conv2(x)
        x=f.sigmoid(x)
        return x
    
input='0_2_90.tif'
model_path='NEW_MODEL256.pt'

model=Model()
checkpoint=torch.load(model_path)
model.load_state_dict(checkpoint)

image=cv2.imread(input, cv2.IMREAD_GRAYSCALE)
image=torch.from_numpy(image)
image=image.unsqueeze(0).float()

device='cuda'

image=image.to(device=device)
model.to(device)

prediction=model(image)
prediction=prediction.detach().cpu().numpy()
# prediction=np.array(prediction.cpu())
save_image=Image.fromarray(prediction)
save_image.save('sharpened_pred')
print("File Saved Successfully")
