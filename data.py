from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self._data = data
        self._mode = mode
        if self._mode == "val":
            # testing/evaluation mode
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])
        else:
            # training mode
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.Resize(size=(350, 350)),
                tv.transforms.RandomCrop(size=300),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        # transform index to a python list
        if torch.is_tensor(index):
            index = index.tolist()

        # read sample information
        image = gray2rgb(imread(Path(self._data.iloc[index, 0])))
        labels = self._data.iloc[index, 1:]
        labels = np.array(labels, dtype=np.float)

        # transform image using the predefined pipeline
        image = self._transform(image)

        # build the complete sample tuple and return it
        return (image, torch.tensor(labels, dtype=torch.float))
