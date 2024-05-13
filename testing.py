from model import DeepLabv3
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

input_img='12478780_15.tiff'
PATH='model.pt'
INPUT_SIZE=256
model=DeepLabv3()

model=torch.load('model.pt')

from PIL import Image
transforms=transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), 2),
        transforms.ToTensor()
    ])

input_image = Image.open(input_img)
input_tensor = transforms(input_image)
input_batch = input_tensor.unsqueeze(0)

output=model(input_batch)

output_predictions_np = output.cpu().numpy()
plt.imshow(output_predictions_np, cmap='gray')
plt.show()