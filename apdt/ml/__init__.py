'''
APDT: Machine Learning
--------------------------

Provide useful modeling and training interface for deep learning.

Check https://github.com/Zhiyuan-Wu/apdt for more information.
'''
from apdt.ml.general import DataSet, TFModel
from apdt.ml.tf_model import wavenet_weight, WaveNet, mlp_weight, MLP, lstm_weight, LSTM
from apdt.ml.tf_model import cnn1d_weight, cnn1d, cnn2d_weight, cnn2d
from apdt.ml.tf_model import Transformer_weight, Transformer
from apdt.ml.tf_model import _WaveNet
from apdt.ml.utils import stacked_window, batch_norm, layer_norm