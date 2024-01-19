import torch
from PIL import Image

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self,item,transform):
        self.item = item
        self.transform = transform

    def __len__(self):
        return len(self.item)

    def __getitem__(self,idx):
        path = self.item[idx].path
        image = Image.open(path).convert('RGB')
        
        image = self.transform(image)
        label = self.item[idx].label
        domain = self.item[idx].domain
        return image,label,domain


class TestDataset(torch.utils.data.Dataset):
    def __init__(self,item,transform):
        self.item = item
        self.transform = transform

    def __len__(self):
        return len(self.item)

    def __getitem__(self,idx):
        path = self.item[idx].path
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        label = self.item[idx].key
        domain = self.item[idx].domain
        return image, label, domain

