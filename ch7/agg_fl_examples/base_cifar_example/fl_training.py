import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from stadle import BasicClient

parser = argparse.ArgumentParser()
parser.add_argument("client_num", type=int)

args = parser.parse_args()
client_num = args.client_num

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
    root='../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2)

# Choose a device based on your machine
device = 'cuda'
# device = 'cpu'

num_epochs = 200
lr = 0.001
momentum = 0.9

model = torchvision.models.vgg16().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr,
                    momentum=momentum, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

parser = argparse.ArgumentParser(description='STADLE CIFAR10 Training')
parser.add_argument('--agent_name', default='default_agent')
args = parser.parse_args()
agent_name = args.agent_name

client_config_path = r'config_agent.json'
stadle_client = BasicClient(config_file=client_config_path, agent_name=agent_name)

for epoch in range(num_epochs):
    if (epoch % 2 == 0):
        # Don't send model at beginning of training
        if (epoch != 0):
            stadle_client.send_trained_model(model)

        state_dict = stadle_client.wait_for_sg_model().state_dict()
        model.load_state_dict(state_dict)

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

        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        sys.stdout.write('\r'+f"\rEpoch Accuracy: {(100*correct/total):.2f}%")
    print('\n')

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