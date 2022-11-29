import tensorflow as tf
from tensorflow import keras
import os
import pickle

class OptimizerCallback(keras.callbacks.Callback):
    '''
    Callback used for periodically saving the state of the optimizer during training.
    '''

    def __init__(self, frequency, save_path) -> None:
        '''
        Args:
            frequency: frequency in epochs
            save_path: location where optimizer weights are saved
        '''
        self.frequency = frequency
        self.save_path = save_path

        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency == 0:
            print("Getting optimizer weights...")
            symbolic_weights = getattr(self.model.optimizer, 'weights')
            weight_values = keras.backend.batch_get_value(symbolic_weights)
            print("Saving optimizer state...")
            with open(os.path.join(self.save_path, 'optimizer_state.pkl'), 'wb') as f:
                pickle.dump(weight_values, f)
            print("Done!")