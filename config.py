#!/usr/bin/env py37
# -*- coding: UTF-8 -*-

#配置


from transformers import BertConfig#自定义 Bert 模型的结构，
# 利用大规模无标注语料训练、获得文本的包含丰富语义信息的Representation即文本的语义表示


class Model_config(BertConfig):

    @classmethod
    def loads(cls, model_config_path):

        model_config = cls.from_pretrained(model_config_path)

        model_config.dataset_name = "BC5CDR"

        model_config.dataset_type = []
        model_config.max_sen_len = 128
        model_config.batch_size = 32
        model_config.id2label = {}
        model_config.label2id = {}
        model_config.num_label = 0
        model_config.dropout = 0.2
        model_config.num_train_epochs = 50
        model_config.device = 'cpu'
        model_config.optim = "AdamW"
        model_config.lr = 5e-5
        model_config.weight_decay = 5e-5
        model_config.seed = 2021

        model_config.output_dir = '../server/fine_tune_model'
        model_config.datasets_dir = '../../data/datasets/data_preprocess/iob_data'


        return model_config
