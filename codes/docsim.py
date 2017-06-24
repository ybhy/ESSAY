#coding:utf-8
#使用docsim方法：doc2bow、similarities判断相似性
from __future__ import division 
from gensim import models,corpora,similarities
import jieba.posseg as pseg
import os
import unicodedata
import sklearn
import numpy as np 
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer


corpora_documents = []
corpora_word = []
for line in open('t6000.txt', 'r').readlines():
	corpora_documents.append(line.strip())
	corpora_word.append(line.split())
# 生成字典和向量语料
print type(corpora_word)
dictionary = corpora.Dictionary(corpora_word) #把所有单词取一个set，并对set中每一个单词分配一个id号的map
corpus = [dictionary.doc2bow(text.split()) for text in corpora_documents]  #把文档doc变成一个稀疏向量，[(0,1),(1,1)]表明id为0,1的词出现了1次，其他未出现。
# print corpus
similarity = similarities.Similarity('Similarity-index', corpus, num_features=len(dictionary) + 1)
size = 6000
length = 1200
preciseAll = 0
print (len(corpora_documents))
for i in xrange(len(corpora_documents)):
	test_doc = corpora_documents[i]
	# print test_doc
	precise = 0
	test_corpus = dictionary.doc2bow(test_doc.split())
	similarity.num_best = length
	for line in similarity[test_corpus]:
		if(i >= 0 and i < 1200):
			if(line[0] >= 0 and line[0] < 1200):
				precise = precise + 1
		elif(i>= 1200 and i < 2400):
			if(line[0] >= 1200 and line[0] < 2400):
				precise = precise + 1
		elif(i>= 2400 and i < 3600):
			if(line[0] >= 2400 and line[0] < 3600):
				precise = precise + 1
		elif(i>= 3600 and i < 4800):
			if(line[0] >= 3600 and line[0] < 4800):
				precise = precise + 1
		else:
			if(line[0] >= 4800 and line[0] < 6000):
				precise = precise + 1
	precise = precise / length
	preciseAll = preciseAll + precise
preciseAll = preciseAll / size
# f.write('topic = ' + str(topics) +'\t' + str(preciseAll) +'\n' )
print preciseAll

	# print(similarity[test_corpus])  # 返回最相似的样本材料,(index_of_document, similarity) tuples