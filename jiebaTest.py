#!/usr/bin/env python
# # coding=utf-8

###加载jieba分词包
import jieba
import jieba.analyse
import jieba.posseg

'''
支持三种分词模式：
精确模式，试图将句子最精确地切开，适合文本分析；
全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
直接分词返回类型generator
'''
print ('jieba分词讲解')
print ('#############################')

test_text = '我今天下午打篮球'

'''
seg_list = jieba.cut(test_text, cut_all=False)#cut_all 参数：是否使用全模式，默认值为 False
seg_list = " ".join(seg_list)
print("cut_all=False:", seg_list)

seg_list2 = jieba.cut(test_text, cut_all=True)
seg_list2 = " ".join(seg_list2)
print("cut_all=True:", seg_list2)

seg_list3 = jieba.cut_for_search(test_text)#cut_for_search 搜索引擎模式
seg_list3 = " ".join(seg_list3)
print("cut_for_search:", seg_list3)


###分词返回类型不同
###########
seg_list = jieba.cut(test_text, cut_all=False)
seg_list4 = jieba.lcut(test_text, cut_all=False)
print("cut返回类型:", type(seg_list))
print("lcut返回类型:", type(seg_list4))


#############HMM隐马尔可夫概率分词模型
seg_list = jieba.cut("他来到了网易杭研大厦", HMM=False)
print("【未启用 HMM】：" + "/ ".join(seg_list))
# 识别新词
seg_list = jieba.cut("他来到了网易杭研大厦") #默认精确模式和启用 HMM
print("【识别新词】：" + "/ ".join(seg_list))



# 繁体字文本
ft_text = "人生易老天難老 歲歲重陽 今又重陽 戰地黃花分外香"
print("【全模式】：" + "/ ".join(jieba.cut(ft_text, cut_all=True)))



####使用txt文档增加自定义新词  文件格式：词语 词频（可省略） 词性（可省略）
sample_text = "周大福是创新办主任也是云计算方面的专家"
print("【未加载词典】：" + '/ '.join(jieba.cut(sample_text)))
jieba.load_userdict('newWord.txt')
print("【加载词典后】：" + '/ '.join(jieba.cut(sample_text)))


###########直接增加自定义词方法add_word
sample_text = "周大福是创新办主任也是云计算方面的专家"
jieba.add_word('创新办主任') #增加自定义词语
print("【增加自定义词】：" + '/ '.join(jieba.cut(sample_text)))
#jieba.add_word('创新办主任', freq=42, tag='nz') #设置词频和词性



####################提取关键词（一段文本中重要具有实际意义或反映文本主题的词）###############


#jieba.analyse.extract_tags()四个参数，使用tfidf定义词语重要性
#sentence：为待提取的文本
#topK：为返回几个 TF/IDF 权重最大的关键词，默认值为 20
#withWeight：是否一并返回关键词权重值，默认值为 False
#allowPOS：仅包括指定词性的词，默认值为空
#################
#前者默认过滤词性（allowPOS=('ns', 'n', 'vn', 'v')）

s = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，  \
    吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。\
    目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"

for x, w in jieba.analyse.extract_tags(s, topK=20, withWeight=True):
    print('%s %s' % (x, w))


'''
#########################返回词性############
words = jieba.posseg.cut("他改变了中国")
for word, flag in words:
    print("{0} {1}".format(word, flag))

