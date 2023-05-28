
import torch
from torch.utils.data import TensorDataset, DataLoader
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class InputExample:
    guid: str
    words: List[str]
    labels: Optional[List[str]]

@dataclass
class InputFeatures:
    token_id: List[int]
    label_id: List[int]
    attention_mask: List[int]
    segment_id: List[int]
    valid_positions:List[int]
    ori_labels_id:List[int]


class NerDataset:

    @classmethod
    def from_iob(cls, filepath):
        # 从IOB文件中读取数据
        # examples中数据为
        # InputExample(guid='train-1',
        #              words=['Immunohistochemical', 'staining', 'was', 'positive', 'for', 'S', '-', '100', 'in', 'all',
        #                     '9', 'cases', 'stained', ',', 'positive', 'for', 'HMB', '-', '45', 'in', '9', '(', '90',
        #                     '%', ')', 'of', '10', ',', 'and', 'negative', 'for', 'cytokeratin', 'in', 'all', '9',
        #                     'cases', 'in', 'which', 'myxoid', 'melanoma', 'remained', 'in', 'the', 'block', 'after',
        #                     'previous', 'sections', '.'],
        #              labels=['O', 'O', 'O', 'O', 'O', 'B-GENE', 'I-GENE', 'I-GENE', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
        #                      'O', 'B-GENE', 'I-GENE', 'I-GENE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
        #                      'O', 'B-GENE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
        #                      'O'])

        with open(filepath, encoding="utf-8") as f:
            guid_index = 1
            examples = []
            words = []
            labels = []
            for line in f:

                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(InputExample(guid=f"train-{guid_index}", words=words, labels=labels))
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split('\t')
                    words.append(splits[0])
                    if len(splits) > 1:
                        splits_replace = splits[-1].replace("\n", "")
                        labels.append(splits_replace)

                    else:
                        labels.append("O")

            if words:
                examples.append(InputExample(guid=f"train-{guid_index}", words=words, labels=labels))

        return examples


    @classmethod
    def get_loader(cls, filepath, tokenizer, max_len, batch_size, label_map):

        examples = cls.from_iob(filepath=filepath)
        print("an examples:", examples[0])

        features = []
        for index, example in enumerate(examples):
            words = example.words
            labels = example.labels

            # 记录下word级别的label。方便计算word级别的准确率
            ori_labels = labels.copy()
            # 超过句子长度把后面去掉
            if len(ori_labels) >= max_len - 1:
                ori_labels = ori_labels[0:(max_len - 2)]
            ori_labels.insert(0, '[CLS]')
            ori_labels.append('[SEP]')
            while len(ori_labels) < max_len:
                ori_labels.append('O')
            assert len(ori_labels) == max_len
            ori_labels_id = [label_map[i] for i in ori_labels]

            # 构造valid_positions，通过subtoken，反推。
            valid_positions = []
            new_tokens = []
            new_labels = []
            for i, word in enumerate(words):
                tokens = tokenizer.tokenize(word)
                label = labels[i]
                new_tokens.extend(tokens)
                for j in range(len(tokens)):
                    if j == 0:
                        valid_positions.append(1)    # 构造valid_positions，通过subtoken，反推。
                        new_labels.append(label)
                    else:
                        valid_positions.append(0)    # 构造valid_positions，通过subtoken，反推。
                        if label != "O":
                            new_labels.append("I-" + label[2:])
                        else:
                            new_labels.append("O")

            if len(new_tokens) > max_len - 2:
                # 这里出现bug，如果直接截断，计算word级别的准确率会出错，be being repaired
                # new_tokens = new_tokens[:max_len - 2]
                # new_labels = new_labels[:max_len - 2]
                # valid_positions = valid_positions[:max_len - 2]
                continue

            # 前加[CLS] 后加[SEP]
            new_tokens.insert(0, '[CLS]')
            new_tokens.append('[SEP]')
            new_labels.insert(0, '[CLS]')
            new_labels.append('[SEP]')
            valid_positions.insert(0, 3)      # 构造valid_positions，通过subtoken，反推。    3 1 1 1 0 0 1 0 1 1 1 0 1 0 1 1 4 -1 -1 -1 -1
            valid_positions.append(4)      # 构造valid_positions，通过subtoken，反推。

            # segment_id和mask
            segment_id = [0] * len(new_tokens)
            mask = [1] * len(new_tokens)

            # 补PAD
            while len(new_tokens) < max_len:
                new_tokens.append('[PAD]')
                new_labels.append('O')
                segment_id.append(0)
                mask.append(0)
                valid_positions.append(-1)     # 构造valid_positions，通过subtoken，反推。

            assert len(new_labels) == len(new_tokens) == len(segment_id) == len(mask) == len(valid_positions) == max_len

            # 转换为id
            token_id = tokenizer.convert_tokens_to_ids(new_tokens)
            label_id = [label_map[i] for i in new_labels]

            features.append(InputFeatures(token_id=token_id,
                                          label_id=label_id,
                                          attention_mask=mask,
                                          segment_id=segment_id,
                                          valid_positions=valid_positions,
                                          ori_labels_id=ori_labels_id))

            if index <= 3:
                print("------------------------an feature example-------------------------------------")
                print("token_ids:", token_id)
                print("label_ids:", label_id)
                print("attention_masks:", mask)
                print("segment_ids:", segment_id)
                print("valid_positions:", valid_positions)
                print("ori_labels_id:", ori_labels_id)

            if len(features) == 100:
                break

        token_ids = torch.tensor([f.token_id for f in features])
        label_ids = torch.tensor([f.label_id for f in features])
        attention_masks = torch.tensor([f.attention_mask for f in features])
        segment_ids = torch.tensor([f.segment_id for f in features])
        valid_positions = torch.tensor([f.valid_positions for f in features])
        ori_labels_id = torch.tensor([f.ori_labels_id for f in features])

        dataset = TensorDataset(token_ids, label_ids, attention_masks, segment_ids, valid_positions, ori_labels_id)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )


        return data_loader, len(features)




