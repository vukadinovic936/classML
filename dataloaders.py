from __future__ import print_function, division

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms


class BinaryDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Image has name and label
        self.images = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.images.iloc[idx, 1])

        img = Image.open(img_name).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = self.images.iloc[idx, 6]

        sample = {'image': img, 'label': label, 'image_path': img_name}

        return sample


class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).
    Note that data is not cloned/copied from the initial dataset.
    """
    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        if self.map:
            x = self.map(self.dataset[index]['image'])
        else:
            x = self.dataset[index]['image']
        y = self.dataset[index]['label']
        z = self.dataset[index]['image_path']
        return x, y, z

    def __len__(self):
        return len(self.dataset)


def get_datasets(csv_file, data_dir,
                     batch_size=1,
                     shuffle_val=False):

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    dataset = BinaryDataset(csv_file=csv_file,
                            root_dir=data_dir)
    train_set, val_set = torch.utils.data.random_split(dataset,
                                                       [int(len(dataset)*80/100), len(dataset) - int(len(dataset)*80/100)])

    transformed_train_set = MapDataset(train_set, data_transforms['train'])
    transformed_val_set = MapDataset(val_set, data_transforms['val'])

    return transformed_train_set, transformed_val_set
