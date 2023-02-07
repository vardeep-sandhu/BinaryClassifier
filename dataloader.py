import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
import torch

def make_dataset(basedir, split):
    img_paths = []
    for (dirpath, _, filenames) in os.walk(basedir):
        if not filenames:
            pass
        else:
            img_paths.extend([join(dirpath, file) for file in filenames])
    # train_size = int(0.8 * len(img_paths))
    # val_size = len(img_paths) - train_size
    # test_size = len(img_paths) - train_size 
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(img_paths, [0.7, 0.15, 0.15])
    if split == "train":
        return train_dataset
    elif split == "test":
        return test_dataset
    elif split == "val":
        return val_dataset

class ImageDataset(Dataset):
    def __init__(self, dataset_path, transform, split):
        self.dataset_path = dataset_path
        self.image_paths = make_dataset(self.dataset_path, split)
        self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        labels = ["valid", "nonvalid"]
        le = preprocessing.LabelEncoder()
        targets = le.fit_transform(labels)
        targets = torch.as_tensor(targets)

        img_path = self.image_paths[idx]
        image = read_image(img_path)
        dir_path = os.path.dirname(img_path)
        # dir_path = "nonvalid/"
        label = os.path.split(dir_path)[-1]  #get directory name as the label 
        assert label in labels, "Check the lable directory."
        
        if label == "nonvalid":
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)
        return image, label

        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # return image, label