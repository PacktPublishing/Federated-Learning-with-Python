import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from stadle import BasicClient

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
# device = 'cuda'
device = 'cpu'

num_rnds = 30
lr = 0.001
momentum = 0.9

model = torchvision.models.vgg16().to(device)
agg_model = torchvision.models.vgg16().to(device)

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


def gamma_inexact_solution_found(curr_grad, agg_grad, gamma):
    if (curr_grad is None):
        return False
    print(curr_grad.norm(p=2), agg_grad.norm(p=2))
    return curr_grad.norm(p=2) < gamma * agg_grad.norm(p=2)


for rnd in range(num_rnds):
    agg_grad = None
    curr_grad = None

    gamma = 0.9
    mu = 0.001

    if (rnd != 0):
        stadle_client.send_trained_model(model)

    sg_model, _ = stadle_client.wait_for_sg_model()
    model.load_state_dict(sg_model.state_dict())
    agg_model.load_state_dict(sg_model.state_dict())

    print('\nRound: %d' % (rnd + 1))

    model = model.to(device)

    model.train()

    while (not gamma_inexact_solution_found(curr_grad, agg_grad, gamma)):
        train_loss = 0
        correct = 0
        total = 0

        total_grad = torch.cat([torch.zeros_like(param.data.flatten()) for param in model.parameters()])

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            curr_weights = torch.cat([param.data.flatten() for param in model.parameters()])
            agg_weights = torch.cat([param.data.flatten() for param in agg_model.parameters()])
            
            prox_term = mu * torch.norm(curr_weights - agg_weights, p=2)**2
            loss += prox_term

            loss.backward()

            grad = torch.cat([param.grad.flatten() for param in model.parameters()])
            total_grad += grad

            optimizer.step()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            sys.stdout.write('\r'+f"\rRound Accuracy: {(100*correct/total):.2f}%")
        print('\n')

        if (agg_grad == None):
            agg_grad = total_grad
        curr_grad = total_grad

    if ((rnd + 0) % 5 == 0):
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