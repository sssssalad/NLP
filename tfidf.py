#!/usr/bin/env python
# coding=utf-8

import jieba
import gensim

#分词并获取词袋函数（为每个出现在语料库中的单词分配一个独一无二的整数编号ID）
#string=['少年强则国强','少年强大则国家强大','少年智则国智']
#string=['傲游AI专注于游戏领域,多年的AI技术积淀,一站式提供文本、图片、\
#音/视频内容审核,游戏AI以及数据平台服务','傲游AI主营游戏领域产品,多年的AI技术沉淀,\
#一站式提供文本、图片、音视频内容审查,游戏AI以及数据服务','傲游AI专注于互联网娱乐领域,\
#多年的人工智能技术积淀,提供全产业链平台及数据服务']
string=['少年强则国强','少年强大则国家强大','少年智则国智','国家强大则少年有为'\
    ,'青少年是祖国的花朵','阴雨天气则会导致运动会推迟','少年时祖国的花朵','中华民族期盼着祖国的强大']
text_list=[]

###分词获取语料库
for sentence in string:
    s_list=[word for word in jieba.cut(sentence)]
    text_list.append(s_list)
#####词典化
dictionary=gensim.corpora.Dictionary(text_list)
print(dictionary)
print(dictionary.token2id)

#向量转换（对每个不同单词出现的次数进行计数并将单词转换为编号，以稀疏向量的形式返回结果）
corpus=[dictionary.doc2bow(doc) for doc in text_list]
print(corpus)

#测试字符串分词并获取词袋函数
#test_string='傲游AI专注于游戏领域,多年的AI技术积淀,一站式提供文本、图片、音/视频内容审核,游戏AI以及数据平台服务'
test_string1 = '少年强则国强'
test_string2 = '少年强大则国家强大'
test_string3 = '少年智则国智'
#test_string1 = '傲游AI专注于游戏领域,多年的AI技术积淀,一站式提供文本、图片、\
#音/视频内容审核,游戏AI以及数据平台服务'
#test_string2 = '傲游AI主营游戏领域产品,多年的AI技术沉淀,\
#一站式提供文本、图片、音视频内容审查,游戏AI以及数据服务'
#test_string3 = '傲游AI专注于互联网娱乐领域,\
#多年的人工智能技术积淀,提供全产业链平台及数据服务'
test_doc_list1=[word for word in jieba.cut(test_string1)]
test_doc_vec1=dictionary.doc2bow(test_doc_list1)
test_doc_list2=[word for word in jieba.cut(test_string2)]
test_doc_vec2=dictionary.doc2bow(test_doc_list2)
test_doc_list3=[word for word in jieba.cut(test_string3)]
test_doc_vec3=dictionary.doc2bow(test_doc_list3)

#使用tfidf模型对语料库建模
tfidf=gensim.models.TfidfModel(corpus)

#分析测试文档与已存在的每个训练样本的相似度
index = gensim.similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
sim1=index[tfidf[test_doc_vec1]]
sim2=index[tfidf[test_doc_vec2]]
sim3=index[tfidf[test_doc_vec3]]
print(sim1)
print(sim2)
print(sim3)
