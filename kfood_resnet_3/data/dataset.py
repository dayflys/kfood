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

        return image,label


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
        return image, label

