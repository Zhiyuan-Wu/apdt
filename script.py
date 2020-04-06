import apdt
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data = apdt.io.gp_data(100000,30)

# Construct dataset
dataset = apdt.ml.DataSet(data, method='window', seq_len=730, normalize_method='max-min')

# Define the model structure
class MyModel(apdt.ml.TFModel):
    def def_model(self, **kwarg):
        self.input = tf.placeholder(tf.float32, shape=(1, 30, 730, 1))
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.pred, self.loss = apdt.ml.WaveNet(self.input[0], apdt.ml.wavenet_weight(), 'wavenet')

# Set up the model
model = MyModel()

# Start Training
model.fit(dataset)