import apdt
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

a = apdt.io.load_weather('1151A','2017-01-01')
b = apdt.io.load_nms(['1151A','1152A'],'2017-01-01')
c = b.merge(a)
c = apdt.proc.spatial_interplot(c,'NN','ALL')

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