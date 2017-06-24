#!/usr/bin/env python
# coding=utf-8  
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

texts = [line.strip() for line in file('ex1000.txt')]
texts_name = [text.split('\t')[0] for text in texts]
# print texts[0]

temp = texts[0].split('\t')
# print len(temp)
# print temp[1]
test_temp = [text.split('\t')[1] for text in texts]
# test_temp = [len(test.split('\t')) for test in texts]
# print test_temp
#引入nltk,文档的单词小写化
texts_lower = [[word for word in document.lower().split()] for document in test_temp]
# print texts_lower
#引入nltk的word_tokenize函数,分离标点符号和单词
texts_tokenized = [[word.lower() for word in word_tokenize(document)] for document in test_temp]
# print texts_tokenized
#过滤停用词,nltk提供的英文停用词数据
english_stopwords = stopwords.words('english')
texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in texts_tokenized]
# print texts_filtered_stopwords
#定义一个标点符号,过滤标点符号
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','"','<','>','-','_']
texts_filtered = [[word for word in document if not word in english_punctuations] for document in texts_filtered_stopwords]
# print texts_filtered
st = LancasterStemmer()
texts_stemmed = [[st.stem(word) for word in docment] for docment in texts_filtered]
all_stems = sum(texts_stemmed, [])
stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
texts_filtered = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]
f = open('t_stem_1000.txt', 'w') 
# print texts_name
size = len(texts_name)
for i in xrange(0, size):
	f.write(texts_name[i] + '\t')
	for word in texts_filtered[i]:
		print word
		f.write(word+' ')
	f.write('\n')
f.close( )
print 'OK'