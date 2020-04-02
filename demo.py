import apdt
import tensorflow as tf

# This script is a demo of apdt
# We will predict air pollution in Beijing NMS by WaveNet.

# Load all sites in china
data = apdt.io.load_nms('ALL','2017-01-01')

# Find sites in Beijing
site_list = apdt.alz.find_sites_range(data, 39.8673, 116.3660, 100)

# Load long term data in these sites
data = apdt.io.load_nms(site_list, '2017-01-01', '2018-12-31')

# Spatial interpolate by IDW first then temporal interpolate
data = apdt.proc.spatial_interplot(data, 'IDW')

# Temporal interpolate by Linear
data = apdt.proc.temporal_interpolate(data, 'ALL', 'linear')

# Construct dataset
dataset = apdt.ml.DataSet(data, method='window', seq_len=730)

# Define the model structure
class MyModel(apdt.ml.TFModel):
    def def_model(self, **kwarg):
        self.input = tf.placeholder(tf.float32, shape=(len(site_list), 730, 1))
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.pred, self.loss = apdt.ml.WaveNet(self.input, apdt.ml.wavenet_weight(), 'wavenet')

# Set up the model
model = MyModel()

# Start Training
model.fit(dataset)