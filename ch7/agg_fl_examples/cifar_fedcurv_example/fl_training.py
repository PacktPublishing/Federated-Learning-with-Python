import sys
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from vgg import VGG

from stadle import BasicClient


def data_processing(data_save_path: str = "./data", max_workers=2, batch_size=64, args=None):

    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

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

    class_counts = int(args.def_count * 5000) * np.ones(len(classes))

    for c in classes:
        if getattr(args, c):
            class_counts[trainset.class_to_idx[c]] = int(args.sel_count * 5000)

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
        trainset, batch_size=batch_size, shuffle=True, num_workers=max_workers)

    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=max_workers)

    return trainloader, testloader


parser = argparse.ArgumentParser(description='STADLE CIFAR10 Training')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

parser.add_argument('--def_count', default=0.1, type=float)
parser.add_argument('--sel_count', default=1.0, type=float)

parser.add_argument('--lt_epochs', default=3)

parser.add_argument('--agent_name', default='default_agent')

parser.add_argument('--cuda', action='store_true', default=False)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

for c in classes:
    parser.add_argument(f'--{c}', action='store_true', default=False)

args = parser.parse_args()

trainloader, testloader = data_processing('../data', args=args)

# Choose a device based on your machine
device = 'cuda' if args.cuda else 'cpu'

num_epochs = 200
lr = 0.001
momentum = 0.9

model = VGG('VGG16').to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr,
                    momentum=momentum, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


fedcurv_lambda = 1.0

fisher_info_params = {f'fedcurv_{c}_{i}':np.zeros(tuple(param.data.size())) for i,param in enumerate(model.parameters()) for c in ('u','v')}
agg_fisher_info_params = None

client_config_path = r'config/config_agent.json'
stadle_client = BasicClient(config_file=client_config_path)

stadle_client.set_bm_obj(model)

for epoch in range(num_epochs):
    if (epoch % 2 == 0):
        # Don't send model at beginning of training
        if (epoch != 0):
            stadle_client.send_trained_model(model, extra_local_params=fisher_info_params)

        sg_model, extra_sg_params = stadle_client.wait_for_sg_model()
        model.load_state_dict(sg_model.state_dict())
        agg_fisher_info_params = extra_sg_params
    
    agg_fisher_info_params = {k:torch.tensor(agg_fisher_info_params[k] - fisher_info_params[k]) for k in fisher_info_params.keys()}

    print('\nEpoch: %d' % (epoch + 1))

    model = model.to(device)

    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        for i,param in enumerate(model.parameters()):
            # Factor out regularization term to use saved fisher info parameters
            reg_term = (param.data ** 2) * agg_fisher_info_params[f'fedcurv_u_{i}']
            reg_term += 2 * param.data * agg_fisher_info_params[f'fedcurv_v_{i}']
            reg_term += (agg_fisher_info_params[f'fedcurv_v_{i}'] ** 2) / agg_fisher_info_params[f'fedcurv_u_{i}']
            loss += fedcurv_lambda * reg_term.sum()

        loss.backward()

        for i,param in enumerate(model.parameters()):
            total_grad[i] += param.grad

        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        sys.stdout.write('\r'+f"\rEpoch Accuracy: {(100*correct/total):.2f}%")
    print('\n')

    for i,param in enumerate(model.parameters()):
        fisher_info_params[f'fedcurv_u_{i}'] = (total_grad[i] ** 2).numpy()
        fisher_info_params[f'fedcurv_v_{i}'] = ((total_grad[i] ** 2) * param.data).numpy()

    if ((epoch + 0) % 5 == 0):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        print(f"Accuracy on val set: {acc}%")

stadle_client.disconnect()