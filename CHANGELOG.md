# Change Log

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
- 发布至pypi