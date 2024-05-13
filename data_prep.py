import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir='Data/Annotations_1024'
output_dir='Data/annot_256'
files=os.listdir(data_dir)

images=[]

for im in files:
    img_path=os.path.join(data_dir, im)
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)


for k in range(len(images)):
    im=images[k]
    x_len=len(im)
    y_len=len(im[0])

    for i in range(0,x_len//256):
        for j in range(0,y_len//256):
            split=img[256*i:256*i+256, 256*j:256*j+256]
            img_save=Image.fromarray(split)
            save_image_path=f'{k}_{256*i+128}_{256*j+128}.tif'
            save_path=os.path.join(output_dir, save_image_path)
            img_save.save(save_path)
