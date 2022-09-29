import torch
from torchvision.models import vgg16

model = vgg16()

torch.save(model, 'saved_vgg_model.pt')