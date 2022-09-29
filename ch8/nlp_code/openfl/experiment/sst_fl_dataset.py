from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from openfl.interface.interactive_api.experiment import DataInterface

import tensorflow as tf

from sst_model import load_sst_data

class SSTShardDescriptor(ShardDescriptor):
    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

        (x_train,y_train), (x_test,y_test) = load_sst_data(self.rank-1, self.worldsize)

        self.data_by_type = {
            'train': tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64),
            'val': tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)
        }

    def get_shard_dataset_types(self):
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train'):
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return self.data_by_type[dataset_type]

    @property
    def sample_shape(self):
        return ["1"]

    @property
    def target_shape(self):
        return ["1"]

    @property
    def dataset_description(self) -> str:
        return (f'SST dataset, shard number {self.rank}'
                f' out of {self.worldsize}')

class SSTFedDataset(DataInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def shard_descriptor(self):
        return self._shard_descriptor

    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        self._shard_descriptor = shard_descriptor
        
        self.train_set = shard_descriptor.get_dataset('train')
        self.valid_set = shard_descriptor.get_dataset('val')

    def get_train_loader(self):
        return self.train_set

    def get_valid_loader(self):
        return self.valid_set

    def get_train_data_size(self):
        return len(self.train_set) * 64

    def get_valid_data_size(self):
        return len(self.valid_set) * 64