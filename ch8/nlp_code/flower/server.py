import flwr as fl

import tensorflow as tf
from tensorflow import keras

from sst_model import SSTModel

MAX_ROUNDS = 3

class SaveKerasModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        agg_weights = super().aggregate_fit(server_round, results, failures)

        if (server_round == MAX_ROUNDS):
            sst_model = SSTModel()
            sst_input = keras.Input(shape=(), dtype=tf.string)
            sst_model(sst_input)
            
            sst_model.set_weights(fl.common.parameters_to_ndarrays(agg_weights[0]))
            sst_model.save('final_agg_sst_model')

        return agg_weights

fl.server.start_server(strategy=SaveKerasModelStrategy(), config=fl.server.ServerConfig(num_rounds=MAX_ROUNDS))