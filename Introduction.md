# APDT 代码文档

这篇文档已被废弃。
This doc has been deprecated.
我们正在将文档迁移到readthedocs。
We are developing a new version powered by readthedocs.

## 1 apdt.general
apdt.general存储一些广泛使用的类型定义。

### 1.1 apdt.general.DataPack
apdt采用DataPack类来存储和处理数据，其有以下属性和方法：

属性：
    
- raw_data: 存储源数据。一个pd.DataFrame，以datetime为索引，每行存储一个时空采样。必需的列包括：lat(纬度)、lon(经度)、data0(第0项数据值)。可选的列包括：site_name(采样点的名称), site_id(采样点的机读ID), data1(第1项数据值)。采样值允许缺失，但时空坐标(datetime, lat, lon)不能有NaN。
- data: 存储用于对外访问的数据。格式同raw_data。
- tag: 存储数据处理标记。一个字符串列表。程序可以检查这一字段来判断数据是否符合某些性质

|tag|描述|来源|
|-|-|-|
|fixed-location|只有固定站数据|数据读取|
|time-aligned|时间采样均匀分布|时间插值、时间重采样|
|non-phisical|不具有直接物理意义|归一化等|

- [Optional] data_type: 记录数据类型。一个字符串列表，分别对应data0、data1等的数据类型
- [Optional] sample_unit: 时间采样单位。可选项包括：'S'(秒), 'H'(小时), 'D'(天).
- [Optional] site_info: 记录站点信息。一个pd.DataFrame，每行存储一个站点信息，列名应与raw_data对应。
- [Optional] normalize_factor：记录归一化因子。一个字典，由'data0'等数据类型映射到(min, max)的数据对。

方法：

- reset()：把数据域重置为新读取状态。
- merge(right, method)：以指定方法合并两个datapack。

### 1.2 apdt.general.SubThread
SubThread是对python的threading类的包装，使其可以方便的读取子线程返回值。

最多可调用的子线程数量由config.yaml中的subthread_max_num字段定义。提供以下方法和属性：

属性：

- func：函数句柄。一个callable。
- args：函数参数。一个tuple。

方法：

- start()：子线程开始运行，这不会阻塞起调进程。
- join()：等待子线程结束，这会阻塞起调进程。
- get_result()：在子线程结束后可以调用，获得子线程的返回值。

## 2 apdt.io
apdt.io处理与数据下载、读取、初始化有关的任务。

### 2.1 apdt.io.load_nms()
该函数读入下载好的NMS (national monitoring stations，国控站) 数据。数据来源为http://beijingair.sinaapp.com/。数据路径由config.yaml中nms_data_path字段指出。国控站数据都是固定站，其输出数据有fixed-location的tag。

其调用方法为：

    Parameters
    ----------
        site: string or list
            The station ID of queried stations like "1001A", or list of such. Sepcifically, if given 'ALL', all available sites will be used.
        start_date: string
            The start time (include) like "2017-01-01".
        end_date: string, default: same as start_date
            The end time (include) like "2018-12-31"
        gas: string or list, default: 'pm2d5'
            The gas type to be collected.
    Return
    ------
        DataPack
            The loaded data that can be used by apdt.

### 2.2 apdt.io.load_weather()
该函数读取由DarkSky网站下载的气象数据。

### 2.3 apdt.io.download_weather()
该函数由DarkSky网站下载气象数据。数据密钥由config.yaml中darksky_secret_key_num和darksky_secret_key_0等字段指明。

请注意由于Darksky网站的免费政策限制，在下载数据量很大时，该函数执行时间会长达数天。

## 3 apdt.ml
apdt.ml提供与深度学习有关的编程工具。

### 3.1 apdt.ml.DataSet
apdt.ml.DataSet自动化的完成数据集的划分、重排和供给。其提供以下属性和方法：

必有属性

- data, 存储原始数据，一个ndarray，其特性视数据集划分方法而不同。
- tr， 存储训练集，一个ndarray，其第0维视为不同样本，dataset将在第零维上将其划分为batch。
- te， 存储测试集，一个ndarray，其第0维视为不同样本，dataset将在第零维上将其划分为batch。
- tr_batch_num，训练集一个epoch内不重复的batch数量。
- te_batch_num，测试集一个epoch内不重复的batch数量。

必有方法

- __init__()：使用一个datapack来进行数据集划分。目前支持的方法有'window'
    - 'window' 即滑窗划分法，该方法只对固定站起作用。在时间轴上划分训练集和测试集，将不同的站点作为一个批次内的不同样本，在时间轴上按照一定的窗长划分为不同的样本。此方法所需的参数表**kwargs为：

|键值|类型|说明|默认值|
|-|-|-|-|
split_ratio|float|训练集所占的比例|0.7
shuffle|bool|是否打乱数据的顺序|True
seq_len|int|batch_size|100
normalize|bool|是否对数据进行归一化|True

    - callable，用户可以自己传入一个数据集划分方法，在接收datapack后，对对象的必有属性赋值。
- tr_get_batch()：从训练集中拿出一个batch。该函数会自动在epoch内循环，自动进行数据打乱和数据供给操作。
- te_get_batch()：从测试集中拿出一个batch。该函数会自动在epoch内循环，自动进行数据打乱和数据供给操作。

### 3.2 apdt.ml.TFModel
apdt.ml.TFModel是Tensorflow模型的基类，提供模型定义、训练、评估有关的功能。其有以下属性和方法：

必有属性：
- input: 用于接收数据输入，TFModel.fit()将调用dataset的get_batch方法，得到的数据将被送入该tensor。形状由用户定义。
- learning_rate：用于接收学习率，TFModel.fit()将自适应地调整学习率。标量。
- pred：网络的输出，形状由用户定义。
- loss：最终用于优化的loss，TFModel.fit()将优化该目标。标量，暂时不支持多目标优化。

必有方法：

- __init__()：初始化函数，默认的初始化函数顺次完成以下操作：设置随机种子、调用def_model构建模型、调用setup_train_op设置优化方式、启动tensorflow的会话、初始化所有参数。
- def_model()：构建模型的函数。必须由用户重载。
- setup_train_op()：设置优化方式的函数。默认状态下会将self.loss送入Adam优化器，并对求得的梯度做截取以防止梯度爆炸，最后将所有梯度作用于所有参数。可选重载。
- fit()：执行模型训练。默认状态下会重复训练若干个epoch，每个epoch内遍历dataset中tr_get_batch所提供的batch，并将其送入input，然后优化loss；此外还会周期性的输出训练日志、保存最优模型、自适应的调整学习率，默认状态下的参数表为：

|键值|类型|说明|默认值|
|-|-|-|-|
model_name|str|模型名称，用于保存模型|NewModel
learning_rate|float|初始学习率|1e-3
baseline|float|基线loss值，优于基线的最好模型将被保存|0(不保存)
epoch|int|训练的epoch数|100
print_every_n_epochs|int|打印训练集损失的间隔epoch数|1
test_every_n_epochs|int|打印测试集损失的间隔epoch数|1
learning_rate_decay_every_n_epochs|int|降低学习率的间隔epoch数|epoch+1（不降低）
learning_rate_decay|float|每次学习率降低的比值1/x|2.0

- update(): 对指定数据优化一次loss，暂未实现。
- eval(): 对指定数据评估输出，暂未实现。

### 3.3 apdt.ml.WaveNet
WaveNet的实现。将一个序列映射到另一个序列作为输出，并返回这两个序列的误差值作为loss。
