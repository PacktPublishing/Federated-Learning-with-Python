import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


# imports from ibmfl lib
from ibmfl.data.data_handler import DataHandler

class CifarDataHandler(DataHandler):
    """
    Data handler for your dataset.
    """
    def __init__(self, data_config=None):
        super().__init__()

        if (data_config is not None):
            if ('client_id' in data_config):
                self.client_id = int(data_config['client_id'])

        self.load_and_preprocess_data()


    def load_and_preprocess_data(self):
        data_save_path = './data'
        def_count = 0.1
        sel_count = 1.0

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
            if c in class_rank_map[self.client_id]:
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

        testset = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True, transform=transform_test)

        self.train_dataset = trainset
        self.val_dataset = testset
    
    def get_data(self):
        """
        Gets the prepared training and testing data.
        
        :return: ((x_train, y_train), (x_test, y_test)) # most build-in training modules expect data is returned in this format
        :rtype: `tuple` 
        """
        return self.train_dataset, self.val_dataset