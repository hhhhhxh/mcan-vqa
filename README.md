[TOC]

# 功能

该程序利用基于注意力机制的深度学习实现了一个视觉问答系统。所谓视觉问答系统，指的是针对给定图片提出的问题作出相应的回答。该程序基于VQA v2.0 Dataset和VG Dataset进行训练，并可以在验证集上进行离线测试，也可以生成结果文件，用于提交到VQA测试平台(https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview)进行在线测试，还可以对某张图片单独输入问题进行询问，并给出回答。

# 文件结构

```
vqa
├── cfgs
│   ├── base_cfgs.py
│   └── path_cfgs.py
├── ckpts
│   └── ckpt_hxh
│       └── epoch13.pkl
├── core
│   ├── data
│   │   ├── ans_punct.py
│   │   ├── answer_dict.json
│   │   ├── data_utils.py
│   │   └── load_data.py
│   ├── model
│   │   ├── mca.py
│   │   ├── net.py
│   │   ├── net_utils.py
│   │   └── optim.py
│		├── exec.py
│		└── show.py
├── results
│   ├── cache
│   │   └── init
│   ├── log
│   │   ├── init
│   │   └── log_run_hxh.txt
│   ├── pred
│   │   └── init
│   └── result_test
│       ├── init
│       └── result_run_hxh_epoch13.json
├── utils
│   ├── proc_ansdict.py
│   ├── vqa.py
│   └── vqaEval.py
├── requirements.txt
├── setup.sh
├── show_pic.py
├── show_test-dev.py
└── run.py
```

/cfgs 存储参数，包括文件路径和运行时的超参数

/ckpts checkpoint，存储训练每个epoch后得到的模型参数，由于模型参数比较大，因此我并未提交

/core/data 数据处理部分

/core/model 网络结构以及优化器部分

/results 记录运行结果

/utils 对VQA的验证部分，参考https://github.com/GT-Vision-Lab/VQA

/run.py 为运行的入口文件

# 数据集

主要用到的数据集为：

1. VQA v2.0 Dataset，下载地址：https://visualqa.org/download.html
2. Visual Genome(VG) Dataset，下载地址：http://visualgenome.org/api/v0/api_home.html
3. MSCOCO Dataset，下载地址：http://cocodataset.org/#download

其中：

VQA v2.0 Dataset和VG Dataset均为VQA数据集，VQA v2.0 Dataset对每张图像都提出了多个问题，每个问题都有10个来自不同的回答。VG Dataset的平均每个图像都有17个QA。并且这两者的图像都来源于MSCOCO的train/val split。

MSCOCO Dataset为图像数据集，用于提取图像特征，我们使用基于Faster R-CNN改进后的bottom-up attention model来对其进行目标检测，并对不同的图像提取10-100个2048维特征。

由于数据集比较大，因此数据集我并未提交，可以在

1. https://github.com/peteanderson80/bottom-up-attention找到coco_extract的下载链接
2. https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EmVHVeGdck1IifPczGmXoaMBFiSvsegA6tf_PqxL3HXclw中下载VG Dataset

然后运行setup.sh，将会下载VQA v2.0 Dataset并解压coco_extract的压缩文件。

数据集的文件结构应如下，其中datasets的路径可在/cfgs/path_cfgs.py中设置。

```
datasets
├─ coco_extract
│  ├─ train2014
│  │  ├── COCO_train2014_...jpg.npz
│  │  └── ...
│  ├─ val2014
│  |  ├── COCO_val2014_...jpg.npz
│  |  └── ...
│  └── test2015
│     ├── COCO_test2015_...jpg.npz
│     └── ...
└── vqa
   ├── v2_OpenEnded_mscoco_train2014_questions.json
   ├── v2_OpenEnded_mscoco_val2014_questions.json
   ├── v2_OpenEnded_mscoco_test2015_questions.json
   ├── v2_OpenEnded_mscoco_test-dev2015_questions.json
   ├── v2_mscoco_train2014_annotations.json
   ├── v2_mscoco_val2014_annotations.json
   ├── VG_questions.json
   └── VG_annotations.json
```

# 超参数

```python
self.GPU = '0'	# 设置在哪个GPU上运行
self.SEED = random.randint(0, 99999999)	# 为CPU和GPU设置随机数种子
self.VERSION = str(self.SEED)	# 设置版本名称
self.RESUME = False	# 是否继续上次的训练

# 在继续训练或测试时必须使用
self.CKPT_VERSION = self.VERSION
self.CKPT_EPOCH = 0

self.CKPT_PATH = None # checkpoint的绝对路径，一旦设置，CKPT_VERSION和CKPT_EPOCH参数会被覆盖
self.VERBOSE = True	# 是否在每步后输出loss

# 数据参数
self.RUN_MODE = 'train'	# {'train', 'val', 'test', 'show'}
self.EVAL_EVERY_EPOCH = True	# Validate(Offline Evaluation)时设置为True
self.TEST_SAVE_PRED = False	# 是否保存prediction vector
self.PRELOAD = False	# 提前把特征加载进内存，可以提高I/O速度

# 定义data split
# 当{'train': 'train'}时EVAL_EVERY_EPOCH为True
self.SPLIT = {
    'train': '',
    'val': 'val',
    'test': 'test',
}

self.TRAIN_SPLIT = 'train+val+vg'	# 额外添加vg数据集作为训练集
self.USE_GLOVE = True	# 是否使用预训练的词向量模型(GloVe: spaCy https://spacy.io/)
self.WORD_EMBED_SIZE = 300	# 词向量矩阵的大小(token size * WORD_EMBED_SIZE)
self.MAX_TOKEN = 14	# 问句的最大长度

# Faster-RCNN得到的2048维特征向量的最大长度
# (Bottom-up and Top-down Attention: https://github.com/peteanderson80/bottom-up-attention)
self.IMG_FEAT_PAD_SIZE = 100
self.IMG_FEAT_SIZE = 2048	# Faster-RCNN 2048维特征向量
self.BATCH_SIZE = 64
self.NUM_WORKERS = 8	# 多线程I/O
self.PIN_MEM = True	# 是否使用PIN_MEMORY(可以加速GPU载入速度但是会提高CPU内存使用)

# 网络参数
self.LAYER = 6	# 模型深度(编码器和解码器的深度相同)
self.HIDDEN_SIZE = 512	# 模型隐层尺寸
self.MULTI_HEAD = 8	# MCA层中Multi-head的数量(HIDDEN_SIZE应该能整除MULTI_HEAD)
self.DROPOUT_R = 0.1	# Dropout rate
self.FLAT_MLP_SIZE = 512	# MLP size in flatten layers

# 使最后一层隐层的变平为有n个attention glimpses的向量
self.FLAT_GLIMPSES = 1
self.FLAT_OUT_SIZE = 1024

# 优化参数
self.LR_BASE = 0.0001	# 基础学习率
self.LR_DECAY_R = 0.2	# 学习率衰减值
self.LR_DECAY_LIST = [10, 12]	# 学习率在第10,12个epoch衰减
self.MAX_EPOCH = 13	# 最大的训练的epoch的数目

# Adam参数
self.OPT_BETAS = (0.9, 0.98)
self.OPT_EPS = 1e-9
```

# 运行结果

运行的模式RUN_MODE分为train, val, test和show四种。其中train为训练，val为验证，test为测试，show为单图片单问题的询问。

RUN_MODE=train时，会在/ckpts下生成名为ckpt_{version}的文件夹，训练完第n个epoch后，会在这个文件夹中生成epoch{n}.pkl的模型参数数据。

RUN_MODE=val时，会在val split上进行测试。

```
Overall Accuracy is: 81.18

Per Answer Type Accuracy is the following:
other : 73.75
yes/no : 95.86
number : 67.05

Write to log file: ./results/log/log_run_first_try_model.txt
```

RUN_MODE=test时，会在/results/result_test下生成result.json文件，为一个list，包含了许多dict，每个dict为一对QA对。可以在https://evalai.cloudcv.org/web/challenges/challenge-page/163/submission上传result文件，等待测试后可以看到accuracy。

```
Result File:
[{"test-dev": {"yes/no": 87.04, "number": 53.22, "other": 60.75, "overall": 70.73}}]

Stdout File:
Preparing global objects..
loading VQA annotations and questions into memory...
0:00:14.539127
creating index...
index created!
Loading and preparing results...     
DONE (t=1.11s)
creating index...
index created!
Evaluating phase..
Elapsed Time: 147.05997943878174
{'test-dev': {'number': 53.22,
              'other': 60.75,
              'overall': 70.73,
              'yes/no': 87.04}}
```

RUN_MODE=show时，可以选择train/val split中的图片，输入图片id和问题，返回结果：

```
========== Loading training set ........
input the image id: 488
488
input the question: What sports are they playing?
What sports are they playing?
========== Finished!
========== Loading ckpt ./ckpts/ckpt_first_try_model/epoch13.pkl
========== Finished!
========== result
[{'question': 'What sports are they playing?', 'answer': 'baseball'}]
```

