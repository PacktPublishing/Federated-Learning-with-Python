import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

from openfl.interface.interactive_api.experiment import TaskInterface
from openfl.interface.interactive_api.experiment import ModelInterface
from openfl.interface.interactive_api.experiment import FLExperiment

from openfl.interface.interactive_api.federation import Federation

from sst_model import get_classification_head, get_sst_full

from sst_fl_dataset import SSTFedDataset

client_id = 'api'
director_node_fqdn = 'localhost'
director_port = 50051

federation = Federation(
    client_id=client_id,
    director_node_fqdn=director_node_fqdn,
    director_port=director_port, 
    tls=False
)

classification_head = get_classification_head()

optimizer = keras.optimizers.Adam(learning_rate=0.005, amsgrad=False)
loss = keras.losses.BinaryCrossentropy()

framework_adapter = 'openfl.plugins.frameworks_adapters.keras_adapter.FrameworkAdapterPlugin'
MI = ModelInterface(model=classification_head, optimizer=optimizer, framework_plugin=framework_adapter)


TI = TaskInterface()

@TI.register_fl_task(model='model', data_loader='train_data', device='device', optimizer='optimizer')
def train(model, train_data, optimizer, device):
    preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    small_bert = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
    small_bert.trainable = False
    
    full_model = get_sst_full(preprocessor, small_bert, model)

    full_model.compile(loss=loss, optimizer=optimizer)
    
    history = full_model.fit(train_data, epochs=1)

    return {'train_loss':history.history['loss'][0]}

@TI.register_fl_task(model='model', data_loader='val_data', device='device')
def validate(model, val_data, device):
    preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    small_bert = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
    small_bert.trainable = False
    
    full_model = get_sst_full(preprocessor, small_bert, model)

    full_model.compile(loss=loss, optimizer=optimizer)

    loss, acc = full_model.evaluate(val_data, batch_size=64)

    return {'val_acc':acc, 'val_loss':loss,}

fed_dataset = SSTFedDataset()

fl_experiment = FLExperiment(federation=federation, experiment_name='sst_experiment')

fl_experiment.start(
    model_provider=MI,
    task_keeper=TI,
    data_loader=fed_dataset,
    rounds_to_train=3,
    opt_treatment='CONTINUE_LOCAL'
)