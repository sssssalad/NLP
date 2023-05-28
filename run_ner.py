#!/usr/bin/env py37
# -*- coding: UTF-8 -*-

import os
import time
import torch
import logging
import torch.nn.functional as F
from collections import Counter

from sklearn.metrics import f1_score, precision_score, recall_score

from src.dev.tokenizer import Tokenizer
from src.fine_tune.model import BNER
from src.datasets.transform import NerDataset
from src.fine_tune.config import Model_config
from src.fine_tune.utils import set_seed, load_label_list

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def main():

    BioBERT_path = "../../data/datasets/ori_biobert/biobert-base-cased-v1.1"
    model_config = Model_config.loads(BioBERT_path)

    # 设置使用设备，gpu-cpu
    model_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 设置种子。默认值2021
    set_seed(model_config.seed)

    # 取出相对应的数据集，'../../data/datasets/data_IOB'
    data_dir = os.path.join(model_config.datasets_dir, model_config.dataset_name)

    # 设置保存模型的路径、log的路径
    save_model_path = os.path.join(model_config.output_dir, model_config.dataset_name)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    save_log_path = os.path.join(save_model_path, model_config.dataset_name + '.txt' )

    # 从训练数据集，构造label_list、dataset_type
    model_config.label_list, model_config.dataset_type = load_label_list(os.path.join(data_dir,'train.txt'))
    model_config.id2label = dict(enumerate(model_config.label_list))
    model_config.label2id = dict(zip(model_config.id2label.values(), model_config.id2label.keys()))
    model_config.num_label = len(model_config.label_list)

    # 加载tokenizer model
    tokenizer = Tokenizer.loads(os.path.join(BioBERT_path, "vocab.txt"), do_lower_case=True)
    model = BNER(BioBERT_path=BioBERT_path, model_config=model_config)
    model.to(model_config.device)

    # 优化器
    optimizer = getattr(torch.optim, model_config.optim)  # config.optim为优化器 为AdamW
    optimizer = optimizer(model.parameters(), lr=model_config.lr, weight_decay=model_config.weight_decay)

    # 读取数据
    train_dataloader, num_train_data = NerDataset.get_loader(
        filepath=os.path.join(data_dir,'train.txt'),
        tokenizer=tokenizer,
        max_len=model_config.max_sen_len,
        batch_size=model_config.batch_size,
        label_map=model_config.label2id
    )

    devel_dataloader, num_devel_data = NerDataset.get_loader(
        filepath=os.path.join(data_dir,'devel.txt'),
        tokenizer=tokenizer,
        max_len=model_config.max_sen_len,
        batch_size=model_config.batch_size,
        label_map=model_config.label2id
    )

    test_dataloader, num_test_data = NerDataset.get_loader(
        filepath=os.path.join(data_dir,'test.txt'),
        tokenizer=tokenizer,
        max_len=model_config.max_sen_len,
        batch_size=model_config.batch_size,
        label_map=model_config.label2id
    )

    with open(save_log_path, 'a') as f:
        f.write("dataset_name:" + str(model_config.dataset_name) + "\n")
        f.write("dataset_type:" + str(model_config.dataset_type) + "\n")
        f.write("num_train_data:" + str(num_train_data) + "\n")
        f.write("num_devel_data:" + str(num_devel_data) + "\n")
        f.write("num_test_data:" + str(num_test_data) + "\n")
        f.write("max_sen_len:" + str(model_config.max_sen_len) + "\n")
        f.write("id2label:" + str(model_config.id2label) + "\n")
        f.write("label2id:" + str(model_config.label2id) + "\n")
        f.write("num_train_epochs:" + str(model_config.num_train_epochs) + "\n")
        f.write("seed:" + str(model_config.seed) + "\n")
        f.write("\n")

    # 开始训练
    for epoch in range(model_config.num_train_epochs):
        train_time_begin = time.time()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(model_config.device) for t in batch)
            token_ids, label_ids, attention_masks, segment_ids, valid_positions, ori_labels_id= batch
            model.zero_grad()
            loss = model(token_ids, attention_masks, segment_ids, label_ids)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print("epoch:{}  | step:{}/{}  | train_loss:{}  | ".format(epoch, step, len(train_dataloader), loss.item()))

        train_time_end = time.time()

        # 验证
        valid_loss, valid_precision, valid_recall, valid_f1 = valid_data(model_config, model, devel_dataloader)

        print("epoch:{}  | cost_time: {} | train_loss:{}|  valid_loss:{}  | valid_precision:{}  | valid_recall:{}  | valid_f1:{}  |".
              format(epoch,
                     round(train_time_end - train_time_begin, 3),
                     round(train_loss/num_train_data, 3),
                     round(valid_loss/num_devel_data, 3),
                     round(valid_precision, 3),
                     round(valid_recall, 3),
                     round(valid_f1, 3)))

        with open(model_config.save_log_path, 'a') as f:
            f.write("epoch:" + str(epoch) + " --|  ")
            f.write("train_loss:" + str(round(train_loss/num_train_data, 3)) + " --|  ")
            f.write("valid_loss:" + str(round(valid_loss/num_devel_data, 3)) + " --|  ")
            f.write("valid_precision:" + str(round(valid_precision, 3)) + " --|  ")
            f.write("valid_recall:" + str(round(valid_recall, 3)) + " --|  ")
            f.write("valid_f1:" + str(round(valid_f1, 3)) + "\n")



    # test
    test_loss, test_precision, test_recal, test_f1 = valid_data(model_config, model, test_dataloader)

    print("test_loss:{}  | test_precision:{}  | test_recal:{}  | test_1:{}  |".
          format(round(test_loss/num_test_data, 3),
                 round(test_precision, 3),
                 round(test_recal, 3),
                 round(test_f1, 3)))

    with open(model_config.save_log_path, 'a') as f:
        f.write("test_result:" + "\n")
        f.write("test_loss:" + str(round(test_loss/num_test_data, 3)) + " --|  ")
        f.write("test_precision:" + str(round(test_precision, 3)) + " --|  ")
        f.write("test_recal:" + str(round(test_recal, 3)) + " --|  ")
        f.write("test_f1:" + str(round(test_f1, 3)) + " --|  ")
        f.write("\n")


    # 保存model，vocab
    torch.save(model, os.path.join(save_model_path, model_config.dataset_name + ".bin"))
    tokenizer.save_vocabulary(save_model_path)
    model_config.to_json_file(os.path.join(save_model_path, "model_config.json"))

def valid_data(model_config, model, data):
    valid_loss = 0.0
    # 验证集验证。
    ori_label_y_pred = []
    ori_label_y_true = []
    with torch.no_grad():
        for step, batch in enumerate(data):
            model.eval()
            batch = tuple(t.to(model_config.device) for t in batch)
            token_ids, label_ids, attention_masks, segment_ids, valid_positions, ori_labels_id = batch
            loss = model(token_ids, attention_masks, segment_ids, label_ids)
            valid_loss += loss.item()

            emissions = model.predict(token_ids, attention_masks, segment_ids)
            logits = F.softmax(emissions, dim=2)
            logits_label = torch.argmax(logits, dim=2)

            att_mast = attention_masks.detach().cpu().numpy().tolist()
            logits_label = logits_label.detach().cpu().numpy().tolist()
            logits_confidence = []
            for i in range(len(logits_label)):
                logits_confidence.append([values[label].item() for values, label in zip(logits[i], logits_label[i])])

            new_label = []
            new_confidence = []
            for i in range(len(logits_label)):
                count = Counter(att_mast[i])[1]
                new_label.append(logits_label[i][:count])
                new_confidence.append(logits_confidence[i][:count])


            # 得到word级别的准确率
            ori_label = ori_labels_id.cpu().numpy().tolist()
            for i in range(len(ori_label)):
                index_sep = ori_label[i].index(model_config.label2id['[SEP]'])
                ori_label_y_true.extend(ori_label[i][:index_sep + 1])  # 算起始和结束位置。0：indenx_4 +1

            valid_positions = valid_positions.cpu().numpy().tolist()
            for i in range(len(valid_positions)):
                index_4 = valid_positions[i].index(4)     # valid_positions = [3 1 1 1 0 0 1 0 1 1 0 0 0 1 1 1 4 -1 -1 -1 -1....]
                valid_i = valid_positions[i][:index_4 + 1]
                y_pred_i = new_label[i]

                new_pred = []
                new_pred.append(y_pred_i[0])
                for index, mask in enumerate(valid_i):
                    if mask == 1:
                        new_pred.append(y_pred_i[index])
                new_pred.append(y_pred_i[-1])
                ori_label_y_pred.extend(new_pred)

        precision_macro, recall_macro, f1_macro = cal(ori_label_y_true, ori_label_y_pred)
    return valid_loss, precision_macro, recall_macro, f1_macro




def cal(y_true,y_pred):
    precision_macro = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    recall_macro = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    return precision_macro,recall_macro,f1_macro



if __name__ == '__main__':

    main()



