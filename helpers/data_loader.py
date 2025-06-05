import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd

FER_MEAN = [0.5]
FER_STD = [0.5]
IMAGE_SIZE = 48

TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=FER_MEAN, std=FER_STD)
])

VAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=FER_MEAN, std=FER_STD)
])

class FER2013Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform or VAL_TRANSFORM 
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pixels = self.data.iloc[idx]['pixels']
        emotion = self.data.iloc[idx]['emotion']
        
        image = np.array([int(pixel) for pixel in pixels.split()]).reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.uint8)
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, emotion


def load_fer_data(csv_path='/content/drive/MyDrive/Assignment4/data/fer2013/fer2013.csv'):
    df = pd.read_csv(csv_path)

    train_df = df[df['Usage'] == 'Training'].copy()
    val_df = df[df['Usage'] == 'PrivateTest'].copy()
    test_df = df[df['Usage'] == 'PublicTest'].copy()

    return train_df, val_df, test_df

def get_fer_dataloaders(train_df, val_df, batch_size=64, num_workers=2):
    train_dataset = FER2013Dataset(train_df, transform=TRAIN_TRANSFORM)
    val_dataset = FER2013Dataset(val_df, transform=VAL_TRANSFORM)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )    
    return train_loader, val_loader

def get_complete_fer_setup(batch_size=64, num_workers=2):
    train_df, val_df, test_df = load_fer_data()
    
    train_loader, val_loader = get_fer_dataloaders(train_df, val_df, batch_size, num_workers)
    
    return train_loader, val_loader, test_df

def get_test_dataloader(test_df, batch_size=64, num_workers=0, transform=VAL_TRANSFORM):
    test_dataset = FER2013Dataset(test_df, transform=transform)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    return test_loader

