import argparse

import tensorflow as tf
from tensorflow import keras

from sst_model import SSTModel, load_sst_data

import flwr as fl

parser = argparse.ArgumentParser()
parser.add_argument("client_id", type=int)

args = parser.parse_args()
client_id = args.client_id

NUM_CLIENTS = 3

(x_train,y_train), (x_test,y_test) = load_sst_data(client_id-1, NUM_CLIENTS)

sst_model = SSTModel()

sst_model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.005, amsgrad=False),
    loss = keras.losses.BinaryCrossentropy(),
    metrics = [keras.metrics.BinaryAccuracy()]
)

sst_input = keras.Input(shape=(), batch_size=64, dtype=tf.string)
sst_model(sst_input)

class SSTClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return sst_model.get_weights()

    def fit(self, parameters, config):
        sst_model.set_weights(parameters)
        history = sst_model.fit(x_train, y_train, epochs=1)
        return sst_model.get_weights(), len(x_train), {'train_loss':history.history['loss'][0]}

    def evaluate(self, parameters, config):
        sst_model.set_weights(parameters)
        loss, acc = sst_model.evaluate(x_test, y_test, batch_size=64)
        return loss, len(x_train), {'val_acc':acc, 'val_loss':loss}

fl.client.start_numpy_client(server_address="[::]:8080", client=SSTClient())