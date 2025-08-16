import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
import pickle
import numpy as np

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

__all__ = ["get_train_loader", "get_val_loader", "CIFAR100Dataset"]

from pathlib import Path
BASE_DATA_PATH = os.path.join(Path(__file__).resolve().parent.parent, "data", "cifar100/")

class CIFAR100Dataset(Dataset):
    def __init__(self, path, transform=None, train=False):
        if train:
            sub_path = 'train'
        else:
            sub_path = 'test'
        with open(os.path.join(path, sub_path), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        with open(os.path.join(path, "meta"), 'rb') as cifar100:
            self.meta_data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

        self.coarse_to_fine_labels = {}
        self.fine_to_coarse_labels = {}
        for i in range(len(self.data['fine_labels'.encode()])):
            fine_label = self.data['fine_labels'.encode()][i]
            coarse_label = self.data['coarse_labels'.encode()][i]
            if fine_label not in self.fine_to_coarse_labels:
                self.fine_to_coarse_labels[fine_label] = coarse_label
            if coarse_label not in self.coarse_to_fine_labels:
                self.coarse_to_fine_labels[coarse_label] = []
            if fine_label not in self.coarse_to_fine_labels[coarse_label]:
                self.coarse_to_fine_labels[coarse_label].append(fine_label)

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        # fine_labels = self.data['fine_labels'.encode()][index]
        # coarse_labels = self.data['coarse_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        fine_labels = torch.tensor(self.data['fine_labels'.encode()][index], dtype=torch.long)
        coarse_labels = torch.tensor(self.data['coarse_labels'.encode()][index], dtype=torch.long)

        return image, fine_labels, coarse_labels

    def len_coarse(self):
        return 20

    def len_fine(self):
        return 100

    def coarse2fine(self):
        return self.coarse_to_fine_labels

    def fine2coarse(self):
        return self.fine_to_coarse_labels




class CIFAR100Test(Dataset):
    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        fine_labels = self.data['fine_labels'.encode()][index]
        coarse_labels = self.data['coarse_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        fine_tensor = torch.tensor(fine_labels, dtype=torch.long)
        coarse_tensor = torch.tensor(coarse_labels, dtype=torch.long)

        fine_one_hot = F.one_hot(fine_tensor, num_classes=100).float().squeeze()
        coarse_one_hot = F.one_hot(coarse_tensor, num_classes=20).float().squeeze()
        return image, fine_one_hot, coarse_one_hot


def get_train_loader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                      transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


def get_val_loader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


def show_batch(images, fine_labels, coarse_labels):
    import matplotlib
    matplotlib.use('TkAgg')
    images = denormalize(images, CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    img_grid = make_grid(images, nrow=4, padding=10, normalize=True)
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.title(f"Fine Labels: {fine_labels}")
    plt.title(f"Coarse Labels: {coarse_labels}")
    plt.show()


def denormalize(tensor, mean, std):
    if not torch.is_tensor(tensor):
        raise TypeError("Input should be a torch tensor.")

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    return tensor

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    # transforms.Grayscale(num_output_channels=1)
    ])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    # transforms.Grayscale(num_output_channels=1)
    ])
batchsize=32
train_dataset = CIFAR100Dataset(path=BASE_DATA_PATH, transform=transform_train,train=True)
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_dataset = CIFAR100Dataset(path=BASE_DATA_PATH, transform=transform_test,train=False)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

if __name__ == "__main__":
    coarse2fine = train_dataset.coarse2fine()
    fine2coarse = train_dataset.fine2coarse()
    print(len(fine2coarse))