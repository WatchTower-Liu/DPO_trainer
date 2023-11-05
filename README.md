# DPO(Direct Preference Optimization)偏好对齐

## 环境说明

代码运行所需的基本环境在requirements.txt中，在控制台运行以下代码安装：

```
pip install -r requirements.txt
```

## 使用

**数据准备**

数据采用如下格式的json文件，是比较常见的配置：


```
{"q":"quary", "a": accept, "r": "reject"}
{"q":"quary", "a": accept, "r": "reject"}
...
```

**配置说明**

训练所需的一些配置在config.yaml中：

```
local_dirs: "/home/liufh/project/data2/liu_project/huggingface_model"  #模型的cache路径
model:
  name_or_path: "/home/liufh/project/data2/liu_project/huggingface_model/qwen/Qwen-14B-Chat"   #模型名或路径

loss:         # DPO训练中的参数，不建议更改
  beta: 0.1
  reference_free: false

lora:
  previous_lora_weights: "../../weights/DPO_test"       # lora模型的路径，如果不使用lora对其则填写空字符串：""

dataset: "../../data/DPO_data_V1.json"            # 对齐所需的数据集
```

**开始训练**

直接运行train.sh即可，支持分布式训练；num_train_epochs推荐为1或2，不宜过大。


