

# Graph_Classification

## 1.代码结构

* graphsn_graph_classification.ipynb 

  GraphSNN cross validation (ipython notebook version)

* graphsn_graph_classification.py

  GraphSNN cross validation (python file version)，这个是自己加的，跟graphsn_graph_classification.ipynb内容一样

* graph_data.py 

  Data preprocessing and loading the data

* data_reader.py 

  Read the txt files containing all data of the dataset

* models.py 

  GNN model with multiple GraphSNN layers for constructing the readout function

* layers.py 

  GraphSNN layer

# 2.graph_data.py

定义了GraphData，用来读取数据，继承了torch.utils.data.Dataset。







# .graphsn_graph_classification.py

使用到的模块：

* `argparse`

  是python用于解析命令行参数和选项的标准模块，`argparse`模块的作用是用于解析命令行参数。

