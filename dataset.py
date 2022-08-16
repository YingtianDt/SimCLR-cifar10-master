from torch.utils.data import Dataset
import cv2
import torch
import os




class ImageDataset(Dataset):
    def __init__(self, data, transform=None, subset_interval=1):
        if subset_interval != 1:
            data = torch.utils.data.Subset(data, list(range(1, len(data), subset_interval)))

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        img, label = self.data[ix]
        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)

        sample1 = {"image": img1, "label": label}
        sample2 = {"image": img2, "label": label}

        return (sample1, sample2)


class ConsistentImageDataset(ImageDataset):
    def __init__(self, data, transform=None, subset_interval=1):
        super().__init__(data, transform, subset_interval)

    def __getitem__(self, ix):
        img, label = self.data[ix]
        if self.transform:
            img = self.transform(img)

        sample1 = {"image": img, "label": label}
        sample2 = sample1

        return (sample1, sample2)