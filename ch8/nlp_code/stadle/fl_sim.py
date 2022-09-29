import argparse

import tensorflow as tf
from tensorflow import keras

from sst_model import SSTModel, load_sst_data

from stadle import BasicClient

parser = argparse.ArgumentParser()
parser.add_argument("client_id", type=int)

args = parser.parse_args()
client_id = args.client_id

NUM_CLIENTS = 3

(x_train,y_train), (x_test,y_test) = load_sst_data(client_id-1, NUM_CLIENTS)

stadle_client = BasicClient(config_file="config_agent.json", agent_name=f"sst_agent_{client_id}")

for round in range(3):
    sst_model = stadle_client.wait_for_sg_model()

    history = sst_model.fit(x_train, y_train, epochs=1)
    loss = history.history['loss'][0]

    stadle_client.send_trained_model(sst_model, {'loss_training': loss})

stadle_client.disconnect()