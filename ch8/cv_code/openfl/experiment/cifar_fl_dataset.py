import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import os
from typing import List

from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from openfl.interface.interactive_api.experiment import DataInterface


# class CifarShardDataset(ShardDataset):
#     def __init__(self, x, y, data_type, rank=1, worldsize=1):
#         self.data_type = data_type
#         self.rank = rank
#         self.worldsize = worldsize
#         self.x = x[self.rank - 1::self.worldsize]
#         self.y = y[self.rank - 1::self.worldsize]

#     def __getitem__(self, index: int):
#         return self.x[index], self.y[index]

#     def __len__(self):
#         return len(self.x)


class CifarShardDescriptor(ShardDescriptor):
    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        
        train_dataset, val_dataset = self.load_cifar_data()
        self.data_by_type = {
            'train': train_dataset,
            'val': val_dataset
        }

    def get_shard_dataset_types(self) -> List[str]:
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train'):
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        
        return self.data_by_type[dataset_type]

    @property
    def sample_shape(self):
        return ["32", "32"]

    @property
    def target_shape(self):
        return ["10"]

    @property
    def dataset_description(self) -> str:
        return (f'Cifar-10 dataset, shard number {self.rank}'
                f' out of {self.worldsize}')

    def load_cifar_data(self, def_count=0.1, sel_count=1.0):
        data_save_path = './data'
        
        classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

        class_rank_map = {
            1: classes[:3],
            2: classes[3:6],
            3: classes[6:]
        }

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=data_save_path, train=True, download=True, transform=transform_train)

        ###
        # Modification for imbalanced train datasets

        class_counts = int(def_count * 5000) * np.ones(len(classes))

        for c in classes:
            if c in class_rank_map[self.rank]:
                class_counts[trainset.class_to_idx[c]] = int(sel_count * 5000)

        class_counts_ref = np.copy(class_counts)

        imbalanced_idx = []

        for i,img in enumerate(trainset):
            c = img[1]
            if (class_counts[c] > 0):
                imbalanced_idx.append(i)
                class_counts[c] -= 1

        trainset = torch.utils.data.Subset(trainset, imbalanced_idx)

        ###

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=64, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=64, shuffle=False, num_workers=2)

        return trainloader, testloader


class CifarFedDataset(DataInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def shard_descriptor(self):
        return self._shard_descriptor

    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        self._shard_descriptor = shard_descriptor
        
        self.train_set = shard_descriptor.get_dataset('train')
        self.valid_set = shard_descriptor.get_dataset('val')

    # def __getitem__(self, index):
    #     return self.shard_descriptor[index]

    # def __len__(self):
    #     return len(self.shard_descriptor)

    def get_train_loader(self):
        return self.train_set

    def get_valid_loader(self):
        return self.valid_set

    def get_train_data_size(self):
        return len(self.train_set)

    def get_valid_data_size(self):
        return len(self.valid_set)