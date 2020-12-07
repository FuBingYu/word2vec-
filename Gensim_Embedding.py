#! usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy
import gensim
import re
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import codecs
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

corpus_file = 'C:/Users/傅冰玉/Desktop/自然语言处理/word2vec进阶版/data/Harry Potter 1.txt'
model_save_file = 'C:/Users/傅冰玉/Desktop/自然语言处理/word2vec进阶版/data/HP1.bin'
embedding_save_file = 'C:/Users/傅冰玉/Desktop/自然语言处理/word2vec进阶版/data/HP.txt'
corpus_new_file = open("C:/Users/傅冰玉/Desktop/Tutorial_4_word2vec-main/data/HP_w.txt","w+")

# 1. Data preprocessing.
sent_list = []
with open(corpus_file)as f:
    for line in f:
        l = line.strip()
        l = re.sub("[!\"#$%&\(\)\*\+,-\./:;<=>@\[\]\\\?^_`\{\}~\|]", '', l)#将所有字符用空格替代
        l = re.sub("\d+\.?\d+", 'NBR', l)#把所有数字用NBR替代
        #l = re.sub("Harry Potter", "HarryPotter", l)
        sent_list.append(l)
    for sent in sent_list:  # 去除文本中的空行
        if sent != '':
            corpus_new_file.write('{0}\n'.format(sent))
        # print(sent_list)

# 2. Define hyper-parameters
size = 100              #词向量的维度
window = 5              #窗口，即词向量上下文最大距离
min_count = 5           #单词出现的最小频率，即忽略词频小于此值的单词
sg = 1                  #模型的训练算法。1：skip-gram，0：CBOW
alpha = 0.02            #初始学习率
epochs = 5              #迭代次数
batch_words = 10000     #每一个batch传递给线程单词的数量。

# 3. Training model
model = gensim.models.Word2Vec(LineSentence(corpus_new_file), size=size, window=window,min_count=min_count,
                               sg=sg, alpha=alpha, iter=epochs,batch_words=batch_words)
#LineSentence(corpus_new_file)：按行读取corpus_new_file文件


# 4. Save Model and Embedding.
# model_save_file.bin文件打开为乱码
model.save(model_save_file)
# embedding_save_file保存了词向量矩阵的shape和所有词的词向量
model.wv.save_word2vec_format(embedding_save_file, binary=False)#nonetype类型

# 5.不用去除停用词
# 拿到了分词后的文件，在一般的NLP处理中，会需要去停用词。
# 打开embedding_save_file文件，可以看到出现了很多停用词。
# 在限制了min_count的情况下，依旧有很多词频很高的停用词出现。
# 但是由于word2vec的算法依赖于上下文，而上下文有可能就是停用词。
# 因此在训练word2vec模型时，我们不能去除停用词，在后续分析的时候，可以视情况而定。

# 6.Use Model
#与Harry最相近的词，topn指定排名前几
top = 95
words = model.wv.most_similar("Harry",topn=top)
print(words)
new_words = []
#与Harry有关的人
for i in range(len(words)):
    pattern = re.compile("[A-Z][a-z]+")
    if re.findall(pattern,words[i][0]) != []:
        new_words.append(words[i][0])
stopword_file = codecs.open("C:/Users/傅冰玉/Desktop//自然语言处理/word2vec进阶版/data/teststopwords.txt", "r", encoding='utf-8')
word_list = []#停用词多行的情况
for word in stopword_file.readlines():
    word_list.extend(word.strip().split(' '))
stopwords = set(word_list)
second_new_words = []
for i in range(len(new_words)):
    if new_words[i] in stopwords:
        continue
    else:
        second_new_words.append(new_words[i])
print(second_new_words)

print("Harry与Ron的相似度：",model.wv.similarity("Harry","Ron"))
#Harry与Ron的相似度： 0.99430805
print("Harry的词向量：",model.wv["Harry"])
'''Harry的词向量： [-0.04646584  0.24373268 -0.11026549 -0.0148825  -0.01014969  0.07843846
  0.02036799 -0.10623636  0.17726895 -0.16243078  0.02186637  0.18282084
 -0.06083002  0.00408263  0.02491822 -0.14466292  0.00222482 -0.03617895
 -0.06825431  0.0395354   0.10162143  0.02555266 -0.03346249  0.09125593
 -0.08543672 -0.18800244  0.1283653  -0.2374186   0.00956841 -0.11194013
  0.11187278  0.08822495 -0.0999632  -0.04470095  0.03632756  0.02027152
 -0.11481963 -0.23698947 -0.16010624 -0.1225024  -0.09129453 -0.1031929
 -0.1184022   0.11255411  0.16647479  0.0128668   0.21716967  0.01045355...'''
print("与Hermione相近的前五个词：",model.wv.similar_by_word("Hermione",topn=5))
#与Hermione相近的前五个词： [('looking', 0.9959883689880371), ('once', 0.9953664541244507),
# ('mouth', 0.9953432083129883), ('followed', 0.9953361749649048), ('pointed', 0.9953173995018005)]

corpus_new_file.close()

