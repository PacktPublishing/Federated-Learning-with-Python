import tensorflow as tf
from tensorflow import keras

from sst_model import SSTModel

sst_model = SSTModel()

optimizer = keras.optimizers.Adam(learning_rate=0.005, amsgrad=False)
loss = keras.losses.BinaryCrossentropy()

sst_model.compile(loss=loss, optimizer=optimizer)

sst_input = keras.Input(shape=(), batch_size=64, dtype=tf.string)
sst_model(sst_input)

sst_model.save('sst_model_save_dir')