# APDT 项目说明

### 1 数据结构

apdt采用DataPack类来存储和处理数据，其有以下属性和方法：

属性：
    
- raw_data: 存储源数据。一个pd.DataFrame，以datetime为索引，每行存储一个时空采样。必需的列包括：lat(纬度)、lon(经度)、data0(第0项数据值)。可选的列建议包括：site_name(采样点的名称), site_id(采样点的机读ID), data1(第1项数据值)
- data: 存储用于对外访问的数据。格式同raw_data。
- tag: 存储数据处理标记。一个字符串列表。
- [Optional] data_type: 记录数据类型。一个字符串列表，分别对应data0、data1等的数据类型
- [Optional] sample_unit: 时间采样单位。可选项包括：'S'(秒), 'H'(小时), 'D'(天).
- [Optional] site_info: 记录站点信息。一个pd.DataFrame，每行存储一个站点信息，列名应与raw_data对应。
