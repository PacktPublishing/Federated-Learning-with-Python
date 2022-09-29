import tensorflow as tf
from tensorflow import keras
from keras import layers

import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text

import numpy as np

class SSTModel(keras.Model):
    def __init__(self):
        super(SSTModel, self).__init__()
        self.preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        self.small_bert = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
        self.small_bert.trainable = False
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        input_dict = self.preprocessor(inputs)
        bert_output = self.small_bert(input_dict)['pooled_output']
        output = self.fc1(keras.activations.relu(bert_output, alpha=0.2))
        scores = self.fc3(self.fc2(output))
        
        return scores

def load_sst_data(client_idx, num_clients=1):
    x_train = []
    y_train = []

    for d in tfds.load(name="glue/sst2", split="train"):
        x_train.append(d['sentence'].numpy())
        y_train.append(d['label'].numpy())

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = []
    y_test = []

    for d in tfds.load(name="glue/sst2", split="validation"):
        x_test.append(d['sentence'].numpy())
        y_test.append(d['label'].numpy())

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    if (client_idx is not None):
        shard_size = int(x_train.size / num_clients)

        x_train = x_train[client_idx*shard_size:(client_idx+1)*shard_size]
        y_train = y_train[client_idx*shard_size:(client_idx+1)*shard_size]

    return (x_train, y_train), (x_test, y_test)