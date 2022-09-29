import tensorflow as tf
from tensorflow import keras

from sst_model import SSTModel, load_sst_data

(x_train,y_train), (x_test,y_test) = load_sst_data()

model = SSTModel

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.0005, amsgrad=False),
    loss = keras.losses.BinaryCrossentropy(),
    metrics = [keras.metrics.BinaryAccuracy()]
)

model.fit(x_train, y_train, batch_size=64, epochs=3)


_, acc = model.evaluate(x_test, y_test, batch_size=64)
print(f"Accuracy of model on test set: {(100*acc):.2f}%")