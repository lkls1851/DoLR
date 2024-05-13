import torch
from model import DeepLabv3
from PIL import Image
from torchvision import transforms
import numpy as np



enter_img_path = '0_2560_512.tif'
enter_model_path = '/media/susmit/OS/Users/susmi/Downloads/massachusset_data/model.pt'



class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1  

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image

model = DeepLabv3()
image = Image.open(enter_img_path)
width_padding = (1024 - image.width) // 2
height_padding = (1024 - image.height) // 2
padded_image = Image.new('RGB', (1024, 1024), color='white')
padded_image.paste(image, (width_padding, height_padding))
padded_image.save("padded_image.jpg")

img_path='padded_image.jpg'
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

dataset = SingleImageDataset(img_path, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

saved_model_dir = enter_model_path
checkpoint = torch.load(saved_model_dir)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

for input, _ in data_loader:
    output = model(input)
    output = torch.argmax(output, dim=1)
    out = Image.fromarray((output.squeeze().cpu().numpy() * 255).astype(np.uint8))
    save_path = 'test_output.jpg'
    out.save(save_path)
