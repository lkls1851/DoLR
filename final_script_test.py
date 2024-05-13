import os

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from model import DeepLabv3
from torch.utils.data import DataLoader
from torchvision import transforms


model_dir='NEW_MODEL.pt'
output_dir='test outputs'
img_path='0_3_90.tif'

INPUT_SIZE=64
model=DeepLabv3()
checkpoint=torch.load(model_dir)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), 2),
        transforms.ToTensor()
    ])



img=Image.open(img_path).convert('RGB')
img=transforms(img)
# img=torch.tensor(img)

device=('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# img= torch.tensor(img, dtype=torch.float32).unsqueeze(0).permute(0,3,1,2).to(device)
# img= torch.tensor(img, dtype=torch.float32).permute(0,3,1,2).to(device)
# img=torch.tensor(img, dtype=torch.float32)
img=torch.tensor(img).unsqueeze(0)
img=img.to(device)

with torch.no_grad():
    output=model(img)
    output=torch.argmax(output, dim=1)

output = output.squeeze(0).cpu().numpy()
output=output*255
output=np.uint8(output)
kernel = np.ones((5,5),np.uint8)
dilated_image = 255-cv2.dilate(255-output, kernel, iterations=1)

out=Image.fromarray((dilated_image).astype(np.uint8))
save_path=os.path.join(output_dir, 'output.jpg')
out.save(save_path)