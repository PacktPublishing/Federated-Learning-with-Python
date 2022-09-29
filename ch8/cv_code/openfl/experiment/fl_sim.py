import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16

from openfl.interface.interactive_api.experiment import TaskInterface
from openfl.interface.interactive_api.experiment import ModelInterface
from openfl.interface.interactive_api.experiment import FLExperiment

from openfl.interface.interactive_api.federation import Federation

from cifar_fl_dataset import CifarFedDataset


client_id = 'api'
director_node_fqdn = 'localhost'
director_port = 50051

federation = Federation(
    client_id=client_id,
    director_node_fqdn=director_node_fqdn,
    director_port=director_port, 
    tls=False
)


model = vgg16()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

framework_adapter = 'openfl.plugins.frameworks_adapters.pytorch_adapter.FrameworkAdapterPlugin'
MI = ModelInterface(model=model, optimizer=optimizer, framework_plugin=framework_adapter)


fed_dataset = CifarFedDataset()

TI = TaskInterface()


@TI.register_fl_task(model='model', data_loader='train_data', device='device', optimizer='optimizer')
def train(model, train_data, optimizer, device):
    for epoch in range(1):  # loop over the dataset multiple times
        model.train()

        for batch_idx, (inputs, targets) in enumerate(train_data):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

    return {}

@TI.register_fl_task(model='model', data_loader='val_data', device='device')
def validate(model, val_data, device):
    model.eval()

    correct = 0
    total = 0
    overall_accuracy = 0

    with torch.no_grad():
        for (inputs, targets) in val_data:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    overall_accuracy = 100 * correct / total

    return {'acc':overall_accuracy,}

fl_experiment = FLExperiment(federation=federation, experiment_name='cifar_experiment')

fl_experiment.start(
    model_provider=MI,
    task_keeper=TI,
    data_loader=fed_dataset,
    rounds_to_train=30,
    opt_treatment='CONTINUE_LOCAL'
)