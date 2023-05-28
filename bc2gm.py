#!/usr/bin/env py37
# -*- coding: UTF-8 -*-
from dataclasses import dataclass
from typing import List
import nltk


@dataclass
class Annotation:
    pmid: str
    term: list
    start: str
    end: str
    type: str


@dataclass
class NerSentence:
    pmid: str
    sentence: str
    annotations: List[Annotation]


def construct_offset_list(full_text):  # 每个句子分词，每个词赋下标
    if full_text[-1] != ' ':
        full_text += ' '
    token_list = []  # 每个句子分词
    offset_list = []  # 分词后的句子中每个词的下标

    word = ''
    offset_index = 0
    offset = 0
    for index, char in enumerate(full_text):
        if char != ' ':
            word += char
            offset = offset + 1
        elif char == ' ':
            token = nltk.word_tokenize(word)
            token_list.extend(token)
            if len(token) == 1:
                offset_list.append(offset_index)
            elif len(token) > 1:
                temp = [offset_index]
                for i, t in enumerate(token):
                    if i != len(token)-1:
                        len_t = len(t)
                        temp.append(offset_index + len_t)
                        offset_index += len_t
                offset_list.extend(temp)
            word = ''
            offset_index = offset

    return token_list, offset_list


class Dataset:

    def to_iob(self, all_NerSentence, output_IOB_filepath):
        for NerSentence in all_NerSentence:
            pmid = NerSentence.pmid
            full_text = NerSentence.sentence
            token_list, offset_list = construct_offset_list(full_text)
            # print(token_list)
            # print(offset_list)
            label_list = ["O"] * len(token_list)  # 首先将每个句子全部赋予标签O

            for annotation in NerSentence.annotations:

                term = annotation.term
                type = annotation.type
                start = int(annotation.start)

                if len(term) == 1:
                    if start in offset_list:
                        index = offset_list.index(start)
                        label_list[index] = "B-" + type  # 如果是实体词，修改标签

                elif len(term) >= 2:
                    len_term = len(term)
                    if start in offset_list:
                        index = offset_list.index(start)
                        label_list[index] = "B-" + type
                        for i in range(1, len_term):
                            label_list[index + i] = "I-" + type

            for token, label in zip(token_list, label_list):  # 最终将提取出的实体词和标签保存在txt文档中
                # 写入txt文件
                with open(output_IOB_filepath, 'a') as f:
                    f.write(token + '\t' + label)
                    f.write("\n")
                    if token == ".":
                        f.write("\n")

    @classmethod
    def from_pubtator(cls, filepath1, filepath2, output_filepath):

        List1 = []
        List2 = []
        all_NerSentence = []
        annotations = []

        with open(filepath1, 'r') as f1, open(filepath2, 'r') as f2:
            for line1 in f1:
                List1.append(line1)
            for line2 in f2:
                List2.append(line2)
            for l1 in List1:
                for l2 in List2:
                    pmid1 = l1.split(' ')[0]
                    pmid2 = l2.replace('|', ' ').split(' ')[0]
                    sentence = l1[len(pmid1) + 1:].replace('\n', '')
                    if pmid1 == pmid2:
                        annotations.append(Annotation(pmid=l2.replace('|', ' ').split(' ')[0],
                                                      term=(l2.replace('|', ' ').replace('\n', '').split(' '))[3:],
                                                      start=l2.replace('|', ' ').split(' ')[1],
                                                      end=l2.replace('|', ' ').split(' ')[2],
                                                      type='Gene'
                                                      ))
                        # print(annotations)
                all_NerSentence.append(NerSentence(pmid=pmid1, sentence=sentence, annotations=annotations))
                annotations = []
            # print(all_NerSentence)
            cls.to_iob(cls, all_NerSentence, output_filepath)

    @classmethod
    def from_xml(cls, filepath, output_filepath):
        pass

    def check(self):
        pass


def gene():
    train_in_filepath = 'ori_data/BC2GM/train.in'
    train_GENE_filepath = 'ori_data/BC2GM/train_GENE.eval'
    train_output_IOB_filepath = 'iob_data/BC2GM/train.txt'

    test_in_filepath = 'ori_data/BC2GM/test.in'
    test_GENE_filepath = 'ori_data/BC2GM/test_GENE.eval'
    test_output_IOB_filepath = 'iob_data/BC2GM/test.txt'

    Dataset().from_pubtator(train_in_filepath, train_GENE_filepath, train_output_IOB_filepath)
    Dataset().from_pubtator(test_in_filepath, test_GENE_filepath, test_output_IOB_filepath)


if __name__ == '__main__':
    gene()
