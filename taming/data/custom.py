
import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
import random

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path



class CustomDataset(Dataset):
    def __init__(self, image_folder, size, transform=None):
        self.data = None
        self.size = size
        self.image_paths = [os.path.join(image_folder, file_name) for file_name in os.listdir(image_folder)]
        self.transform = transform

        


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        # Charger l'image
        image = Image.open(self.image_paths[i])
        to_tensor = transforms.Compose([
            #transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.size, self.size)),  # Correction ici
            transforms.ToTensor()
        ])
        image = to_tensor(image)

        return {"image": image }



class CustomDatasetCrop(Dataset):
    def __init__(self, size, image_folder, transform=None, random_crops=0, num_channels=3):
        self.image_paths = [os.path.join(image_folder, file_name) for file_name in os.listdir(image_folder)]
        self.transform = transform
        self.random_crops = random_crops
        self.taille_crop = size // 8
        self.num_crop = 8 * 8  # Nombre total de crops fixes
        self.num_channels = num_channels

    def __getitem__(self, index):
        # Charger l'image
        
        #print("self.image_paths[index]", self.image_paths)
        image = Image.open(self.image_paths[index])
        
        # Redimensionner l'image à 1024x1024
        image = image.resize((1024, 1024))
        
        # Extraction de crops fixes
        crops = [image.crop((i % 8 * self.taille_crop, i // 8 * self.taille_crop, 
                             (i % 8 + 1) * self.taille_crop, (i // 8 + 1) * self.taille_crop))
                 for i in range(self.num_crop)]
        
        # Ajout des crops aléatoires
        for _ in range(self.random_crops):
            x = random.randint(0, image.width - self.taille_crop)
            y = random.randint(0, image.height - self.taille_crop)
            random_crop = image.crop((x, y, x + self.taille_crop, y + self.taille_crop))
            crops.append(random_crop)

        # Transformation des crops
        if self.transform:
            crops = [self.transform(crop) for crop in crops]
        else:
            
            to_tensor = transforms.Compose([
                #transforms.Grayscale(num_output_channels=self.num_channels),
                transforms.ToTensor()
            ])
            crops = [to_tensor(crop) for crop in crops]

        # Empilement des tensors de crops
        crops_tensor = torch.stack(crops)

        return {"image": crops_tensor, "path": self.image_paths[index] }

    def __len__(self):
        return len(self.image_paths)




class CustomTrain(CustomDataset):
    def __init__(self, image_folder, size, transform=None):
        super().__init__(image_folder,size, transform)
        
        for root, _, files in os.walk(image_folder):
            for file in files:

                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))



class CustomTest(CustomDataset):
    def __init__(self, image_folder, size,  transform=None):

        super().__init__(image_folder, size, transform)
        
        for root, _, files in os.walk(image_folder):
            for file in files:

                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))



class CustomTrain_crop(CustomDatasetCrop):
    def __init__(self, size, image_folder, transform=None, random_crops=0):
        super().__init__(size, image_folder, transform, random_crops)
        
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))


import os

class CustomTest_crop(CustomDatasetCrop):
    def __init__(self, size, image_folder, transform=None, random_crops=0, num_channels=3):
        super().__init__(size, image_folder, transform, random_crops, num_channels)
        
        # Walk through the directory structure starting at 'image_folder'
        for root, dirs, files in os.walk(image_folder):
            for file in files:
                file_path = os.path.join(root, file)

                # Check if the path is indeed a file and not a directory
                if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    # Append the full path of the file to image_paths
                    self.image_paths.append(file_path)

       

