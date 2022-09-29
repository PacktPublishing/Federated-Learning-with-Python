import tensorflow as tf
from tensorflow import keras

from sst_model import SSTModel, load_sst_data

import nest_asyncio
nest_asyncio.apply()
import tensorflow_federated as tff

NUM_CLIENTS = 1
NUM_ROUNDS = 3

# Load three disjoint subsets, representing client datasets
client_datasets = [load_sst_data(idx, NUM_CLIENTS)[0] for idx in range(NUM_CLIENTS)]

def sst_model_fn():
    sst_model = SSTModel()
    sst_model.build(input_shape=(None,64))

    return tff.learning.from_keras_model(
        sst_model,
        input_spec=tf.TensorSpec(shape=(None), dtype=tf.string),
        loss=keras.metrics.BinaryCrossentropy()
    )

fed_avg_process = tff.learning.algorithms.build_unweighted_fed_avg(
    model_fn = sst_model_fn,
    client_optimizer_fn = lambda: keras.optimizers.Adam(learning_rate=0.005, amsgrad=False),
    server_optimizer_fn = lambda: keras.optimizers.SGD(learning_rate=1.0)
)

state = fed_avg_process.initialize()

for round in range(NUM_ROUNDS):
    state = fed_avg_process.next(state, client_datasets).state

fed_weights = fed_avg_process.get_model_weights(state)

fed_sst_model = SSTModel()
fed_sst_model.build(input_shape=(None, 64))
fed_sst_model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.005, amsgrad=False),
    loss = keras.losses.BinaryCrossentropy(),
    metrics = [keras.metrics.BinaryAccuracy()]
)

fed_weights.assign_weights_to(fed_sst_model)

_, (x_test, y_test) = load_sst_data()
_, acc = fed_sst_model.evaluate(x_test, y_test, batch_size=64)

print(f"Accuracy of federated model on test set: {(100*acc):.2f}%")