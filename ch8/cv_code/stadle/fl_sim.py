import os
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16

import flwr as fl


data_save_path = './data'
def_count = 0.2
sel_count = 1.0

parser = argparse.ArgumentParser()
parser.add_argument("client_num", type=int)

args = parser.parse_args()
client_num = args.client_num

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
    if c in class_rank_map[client_num]:
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


model = vgg16()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        model.train()

        for epoch in range(1):  # loop over the dataset multiple times
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                print(batch_idx)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                print(loss.item())
                loss.backward()
            optimizer.step()

        return self.get_parameters({}), len(trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        model.eval()

        correct = 0
        total = 0
        overall_accuracy = 0

        with torch.no_grad():
            for (inputs, targets) in testloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        overall_accuracy = 100 * correct / total

        print(overall_accuracy)

        return 0.0, len(testloader), {'acc': overall_accuracy}
        

fl.client.start_numpy_client(server_address="[::]:8080", client=CifarClient())