import os

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

# Assuming 'model' is your PyTorch model
model = torch.load('model.pt')
model.eval()

output_dir = 'test_outputs'
os.makedirs(output_dir, exist_ok=True)

img_path = '12478780_15.tiff'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)

output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

out = Image.fromarray((output * 255).astype(np.uint8))
save_path = os.path.join(output_dir, 'output.jpg')
out.save(save_path)