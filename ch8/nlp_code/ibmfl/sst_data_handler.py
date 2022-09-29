from ibmfl.data.data_handler import DataHandler

import tensorflow as tf

from sst_model import load_sst_data

class SSTDataHandler(DataHandler):
    def __init__(self, data_config=None):
        super().__init__()

        if (data_config is not None):
            if ('client_id' in data_config):
                self.client_id = int(data_config['client_id'])
            if ('num_clients' in data_config):
                self.num_clients = int(data_config['num_clients'])

        train_data, val_data = load_sst_data(self.client_id-1, self.num_clients)

        self.train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(64)
        self.val_dataset = tf.data.Dataset.from_tensor_slices(val_data).batch(64)
    
    def get_data(self):
        return self.train_dataset, self.val_dataset