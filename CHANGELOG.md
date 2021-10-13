# Change Log

#### V0.1.8
- ml.TFModel 增加 1-D CNN
- ml.TFModel 增加 2-D CNN
- 增加 ml.batch_norm 和 ml.layer_norm 方法
- 重做 TFModel.Transformer
- 为 MLP 增加 bn 支持
- ml.DataSet 增加 shuffle_train_only 参数
- TFModel 支持定义多个 loss
- TFModel 支持重载 update 方法构造复杂的优化过程
- TFModel 增加优化器选择：Adam, RMSProp
- TFModel 增加 iter_do 方法，在优化过程中执行指定操作

#### V0.1.7
- 增加 alz.moving_prediction 方法
- ml.WaveNet 现在返回和输入序列长度相同的特征
- ml.WaveNet 增加 valid_loss 参数，开启后只在有效观测区间计算loss
- 增加 ml.stacked_window 方法
- TFModel 可以使用 tf.summary 来自动启用TensorBoard
- TFModel 支持以 list 作为 self.metric
- TFModel 增加 self.training_process 量指示训练进程

#### V0.1.6
- 增加 io.DarkSkyAPI
- ml.TFModel 中 clip_grad 默认改为 False
- ml.TFModel.fit 增加 higher_better 参数
- io.load_nms 增加关闭并行选项
- ml.Dataset 支持有监督学习模式
- ml.Dataset 支持直接读入MNIST
- ml.TFModel 增加 LSTM
- ml.TFModel 增加 Transformer

#### V0.1.5
- TFModel支持预训练参数自动载入
- TFModel预训练参数支持从多个模型分别载入

#### V0.1.4
- Dataset支持训练、验证、测试三级划分，同时兼容训练、测试二级划分。
- TFModel按验证最优报告测试性能
- TFModel默认总是保存模型
- TFModel支持自动重复训练，计算置信区间
- TFModel支持自定义不同的优化目标和评价指标
- alz增加mcl算法，一种简易的无监督图聚类算法

#### V0.1.3
- WaveNet 增加BN层
- TFModel增加全局训练/测试指示变量
- DataPack增加dump/load方法

#### V0.1.2
- 重写WaveNet，训练/推断速度大幅提升，移除了对输入序列长度的限制，更灵活的参数输入方式。

## V0.1

- 重新初始化apdt包，设alz, io, ml, plt, proc, general六个子模块；
- general中增加DataPack和SubThread类；
- alz中增加haversine, IDW, find_sites_range函数；
- io中增加国控站数据load_nms函数；
- 加入基于高斯过程的仿真数据生成方法io.gp_data；
- 加入对佛山固定站数据的初步支持；
- 加入基于Darksky API的实时气象数据支持；
- ml中增加DataSet和TFModel类；
- ml中增加WaveNet的模型定义；
- ml中增加MLP的模型定义；
- proc中增加spatial_interplot, temporal_interpolate两大插值过程；
- 加入基于小波的事件监测方法alz.wavelet_detect（测试）；
- 初始化项目文档Introduction.md，但格式仍需调整；
- 初始化变更日志CHANGELOG.md；
- 构建script.py，以北京PM2.5预测为demo；
