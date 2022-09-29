from collections import OrderedDict

import flwr as fl

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg16

MAX_ROUNDS = 30

class SavePyTorchModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        agg_weights = super().aggregate_fit(server_round, results, failures)

        if (server_round == MAX_ROUNDS):
            vgg_model = vgg16()
            
            np_weights = fl.common.parameters_to_ndarrays(agg_weights[0])
            params_dict = zip(vgg_model.state_dict().keys(), np_weights)

            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            
            torch.save(state_dict, "final_agg_vgg_model.pt")

        return agg_weights

fl.server.start_server(
    strategy=SavePyTorchModelStrategy(),
    config=fl.server.ServerConfig(num_rounds=MAX_ROUNDS),
    grpc_max_message_length=1024**3
)