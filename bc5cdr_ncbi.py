#!/usr/bin/env py37
# -*- coding: UTF-8 -*-

from dataclasses import dataclass#适合于存储数据对象的Python类，把一个类转化为dataclass装饰器
from typing import List#用于类型注解(type hints)的模块，为python程序提供静态类型检查
import nltk#引入⾃然语⾔处理库


@dataclass
class Annotation:#注解的类
    pmid: str
    term: str
    start: str
    end: str
    type: str


@dataclass
class NerArticle:#文章的类
    pmid: str
    title: str
    abstract: str
    annotations: List[Annotation]

def construct_offset_list(full_text):
    if full_text[-1] != ' ':
        full_text += ' '

    token_list = []
    offset_list = []

    word = ''
    offset = 0
    for index, char in enumerate(full_text):
        if char != ' ':
            word += char
        elif char == ' ':
            token = nltk.word_tokenize(word)
            token_list.extend(token)
            if len(token) == 1:
                offset_list.append(offset)
            elif len(token) > 1:
                temp = [offset]
                for i, t in enumerate(token):
                    if i != len(token) - 1:
                        len_t = len(t)
                        temp.append(offset + len_t)
                        offset += len_t
                offset_list.extend(temp)
            word = ''
            offset = index + 1

    return token_list, offset_list





class Dataset:

    def to_iob(self, all_NerArticle, output_IOB_filepath):
        for article in all_NerArticle:
            pmid = article.pmid
            full_text = article.title + " " + article.abstract
            token_list, offset_list = construct_offset_list(full_text)
            label_list = ["O"] * len(token_list)

            for annotation in article.annotations:

                term = annotation.term
                type = annotation.type
                start = int(annotation.start)

                term_token = nltk.word_tokenize(term)

                if len(term_token) == 1:
                    if start in offset_list:

                        index = offset_list.index(start)
                        label_list[index] = "B-" + type

                elif len(term_token) >= 2:
                    len_term_token = len(term_token)
                    if start in offset_list:
                        index = offset_list.index(start)
                        label_list[index] = "B-" + type
                        for i in range(1, len_term_token):
                            label_list[index + i] = "I-" + type

            for token, label in zip(token_list, label_list):
                # 写入csv文件
                with open(output_IOB_filepath, 'a') as f:
                    f.write(token + '\t' + label)
                    f.write("\n")
                    if token == ".":
                        f.write("\n")


    @classmethod
    def from_pubtator(cls, filepath, output_filepath):

        label_list = []

        all_article = []

        pmid = -1
        title = ''
        abstract = ''
        annotations = []


        with open(filepath, 'r') as f:


            for line in f:
                # 判断每一行是title还是abstarct或者annotation
                if not line.strip():
                    all_article.append(NerArticle(pmid=pmid,title=title,abstract=abstract,annotations=annotations))
                    pmid = -1
                    title = ''
                    abstract = ''
                    annotations = []
                else:
                    line_elements = line.split('|')
                    if len(line_elements) == 1:

                        annotation_elements = line_elements[0].split('\t')
                        if annotation_elements[1] == 'CID':
                            continue
                        else:
                            if annotation_elements[4] not in label_list:
                                label_list.append(annotation_elements[4])
                            annotations.append(Annotation(pmid=annotation_elements[0],
                                                         term=annotation_elements[3],
                                                         start=annotation_elements[1],
                                                         end=annotation_elements[2],
                                                         type=annotation_elements[4]
                                                         ))
                    elif line_elements[1] == 't':
                        pmid = line_elements[0]
                        title = line_elements[2].split('\n')[0]

                    elif line_elements[1] == 'a':
                        abstract = line_elements[2].split('\n')[0]

            cls.to_iob(cls, all_article, output_filepath)
            print("label_list:",label_list)

    @classmethod
    def from_xml(cls, filepath, output_filepath):


        pass

    def check(self):
        pass


def bc5cdr():
    #导入训练集
    #train_filepath = 'ori_data/BC5CDR/CDR_TrainingSet.PubTator.txt'
    #train_output_IOB_filepath = 'iob_data/BC5CDR/train.txt'
    train_filepath = 'D:\CDR_TrainingSet.PubTator.txt'
    train_output_IOB_filepath = 'D://train.txt'

    #导入开发集
    # devel_filepath = 'ori_data/BC5CDR/CDR_DevelopmentSet.PubTator.txt'
    # devel_output_IOB_filepath = 'iob_data/BC5CDR/devel.txt'
    devel_filepath = 'D:/CDR_DevelopmentSet.PubTator.txt'
    devel_output_IOB_filepath = 'D://devel.txt'

    #导入测试集
    # test_filepath = 'ori_data/BC5CDR/CDR_TestSet.PubTator.txt'
    # test_output_IOB_filepath = 'iob_data/BC5CDR/test.txt'
    devel_filepath = 'D:/CDR_DevelopmentSet.PubTator.txt'
    devel_output_IOB_filepath = 'D://devel.txt'

    Dataset().from_pubtator(train_filepath, train_output_IOB_filepath)
    Dataset().from_pubtator(devel_filepath, devel_output_IOB_filepath)
    Dataset().from_pubtator(test_filepath, test_output_IOB_filepath)

# def ncbi_disease():
#
#     train_filepath = 'ori_data/NCBI-disease/NCBItrainset_corpus.txt'
#     train_output_IOB_filepath = 'iob_data/NCBI-disease/train.txt'
#
#     devel_filepath = 'ori_data/NCBI-disease/NCBIdevelopset_corpus.txt'
#     devel_output_IOB_filepath = 'iob_data/NCBI-disease/devel.txt'
#
#     test_filepath = 'ori_data/NCBI-disease/NCBItestset_corpus.txt'
#     test_output_IOB_filepath = 'iob_data/NCBI-disease/test.txt'
#
#     Dataset().from_pubtator(train_filepath, train_output_IOB_filepath)
#     Dataset().from_pubtator(devel_filepath, devel_output_IOB_filepath)
#     Dataset().from_pubtator(test_filepath, test_output_IOB_filepath)



def jnlpba():

    pass

def bc2gm():

    pass

if __name__ == '__main__':
    """
        BC5CDR和NCBI-disease的原始数据是相同的格式。
        BC5CDR：['Chemical', 'Disease']
        NCBI-disease：['DiseaseClass', 'SpecificDisease', 'Modifier', 'CompositeMention']
    """
    bc5cdr()
    #ncbi_disease()
    # JNLPBA()
