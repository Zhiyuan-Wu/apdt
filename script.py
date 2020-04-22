import apdt
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
gas_type = 'o3'
if os.path.exists('data/nanjing'+gas_type+'.pkl'):
    dataset = apdt.ml.DataSet('data/nanjing'+gas_type+'.pkl')
else:
    site_list = ['115'+str(i)+'A' for i in range(1,10)]
    feature_list = ['wendu', 'shidu', 'fengsu', 'fengxiang', 'qiya']
    a = apdt.io.load_weather('1151A','2017-01-01','2020-04-15',feature=feature_list)
    b = apdt.io.load_nms(site_list,'2017-01-01','2020-04-15',gas=gas_type)
    c = b.merge(a)
    c = apdt.proc.spatial_interplot(c,'IDW','ALL')
    c = apdt.proc.temporal_interpolate(c)

    # Construct dataset
    dataset = apdt.ml.DataSet(c, method='window', seq_len=730, normalize_method='999pt')
    print(c.normalize_factor)
    dataset.dump('data/nanjing'+gas_type+'.pkl')

# Define the model structure
class MyModel(apdt.ml.TFModel):
    def def_model(self, **kwarg):
        self.input = tf.placeholder(tf.float32, shape=(1, 9, 730, 6))
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.pred, self.loss = apdt.ml.WaveNet(self.input[0], apdt.ml.wavenet_weight(input_dim=6), 'wavenet')

# Set up the model
model = MyModel()

# Start Training
model.fit(dataset, baseline=10.0, epoch=300)
