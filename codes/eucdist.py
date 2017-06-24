#!/usr/bin/env python
# coding=utf-8  
#读取tfidf值的文档，向量化后计算similarity

import time          
import re          
import os  
import sys
import codecs
import random
import shutil
import numpy
import numpy as np
from numpy import *
from numpy import arange
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer 
#························计算tfidf·························
corpus = []
for line in open('text3000.txt', 'r').readlines():
	line = line.split('\t')
	corpus.append(line[1].strip())
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
word = vectorizer.get_feature_names()
weight = tfidf.toarray()
print type(weight)
# print 'Features length: ' + str(len(word))
print len(weight)
print len(weight[0])
# print weight
resName = "eucdist3000.txt"
result = codecs.open(resName, 'w', 'utf-8')
for i in xrange(0, len(weight)):
	for j in xrange(0, len(weight)):
		if(i != j):
			xi = weight[i]
			xj = weight[j]
			# print (xi - xj)
			euc = numpy.sqrt(numpy.sum(numpy.square(xi - xj)))  
			# cos= dot(xi,xj)/(linalg.norm(xi)*linalg.norm(xj))
			# similarity = - np.sqrt(np.dot(xi - xj, np.transpose(xi - xj)))
			record = "%d %d %f " % (i,j,euc)
			result.write(record)
			result.write('\r\n')
result.close()