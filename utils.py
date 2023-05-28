#!/usr/bin/env py37
# -*- coding: UTF-8 -*-




import torch
import random
import numpy as np
import logging#保存程序运行的日志，以排查程序在某一个时候崩溃的具体原因，记录程序运行的过程的中的一些信息

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


# 设置种子
def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_label_list(label_list_path):
    label_list = ['[PAD]', '[CLS]', '[SEP]', 'O']
    dataset_type = []
    with open(label_list_path, 'r') as f:
        for line in f:
            if line.strip():
                label = line.split('\t')[1].split("\n")[0]
                if label not in label_list:
                    label_list.append(label)

                if label[2:] not in dataset_type and label != "O":
                    dataset_type.append(label[2:])

    return label_list,dataset_type
