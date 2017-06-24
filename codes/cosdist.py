#!/usr/bin/env python
# coding=utf-8  
#读取tfidf值的文档，向量化后计算similarity
from __future__ import division
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
for line in open('t1000.txt', 'r').readlines():
	line = line.split('\t')
	corpus.append(line[1].strip())
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
print tfidf
word = vectorizer.get_feature_names()
print word
weight = tfidf.toarray()
print weight
print type(weight)
# print 'Features length: ' + str(len(word))
print len(weight)
print len(weight[0])
# print weight
resName = "cosdist3000.txt"
result = codecs.open(resName, 'w', 'utf-8')
for i in xrange(0, len(weight)):
	for j in xrange(0, len(weight)):
		if(i != j):
			xi = weight[i]
			xj = weight[j]
			# intersection = 0
			# union = 0
			# for k in xrange(0, len(xi)):
			# 	if xi[k] > 0 and xj[k] > 0:
			# 		intersection = intersection + 1
			# 	if xi[k] > 0 or xj[k] > 0:
			# 		union = union + 1
			# if union == 0:
			# 	jaccard = 0#!/usr/bin/env python
# coding=utf-8  
#读取tfidf值的文档，向量化后计算similarity
from __future__ import division
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
for line in open('ex1000.txt', 'r').readlines():
	line = line.split('\t')
	corpus.append(line[1].strip())
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
# print tfidf
word = vectorizer.get_feature_names()
# print word
weight = tfidf.toarray()
# print weight
print type(weight)
# print 'Features length: ' + str(len(word))
print len(weight)
print len(weight[0])
# print weight
resName = "cosdist3000.txt"
result = codecs.open(resName, 'w', 'utf-8')
for i in xrange(0, len(weight)):
	for j in xrange(0, len(weight)):
		if(i != j):
			xi = weight[i]
			xj = weight[j]
			intersection = 0
			union = 0
			for k in xrange(0, len(xi)):
				if xi[k] > 0 and xj[k] > 0:
					intersection = intersection + 1
				if xi[k] > 0 or xj[k] > 0:
					union = union + 1
			if union == 0:
				jaccard = 0
			else:
				jaccard = intersection / union

			print "%d %d %f" % (i, j, jaccard)

# 			# print (xi - xj)
# 			# dist = numpy.sqrt(numpy.sum(numpy.square(xi - xj)))  
# 			cos= dot(xi,xj)/(linalg.norm(xi)*linalg.norm(xj))
# 			# similarity = - np.sqrt(np.dot(xi - xj, np.transpose(xi - xj)))
# 			record = "%d %d %f " % (i,j,cos)
# 			# result.write(record)
# 			# result.write('\r\n')
# result.close()

	sort_cos = sorted(d.iteritems(), key=lambda d:d[1], reverse = False)
	# print sort_euc
	precise = 0
	# for (k,v) in sort_euc.items():
	for line in sort_cos[0:length]:
		k = line[0]
		# print k
		if i < 1000:
			if(k >= 0 and k < 1000):
				precise = precise + 1
		elif i <  2000:
			if(k >= 1000 and k < 2000):
				precise = precise + 1
		else:
			if(k >= 2000 and k < 3000):
				precise = precise + 1
	precise = precise / length
	preciseAll = preciseAll + precise
	d.clear()
preciseAll = preciseAll / size
print preciseAll
			# else:
			# 	jaccard = intersection / union

			# print "%d %d %f" % (i, j, jaccard)

			# print (xi - xj)
			# dist = numpy.sqrt(numpy.sum(numpy.square(xi - xj)))  
			cos= dot(xi,xj)/(linalg.norm(xi)*linalg.norm(xj))
			# similarity = - np.sqrt(np.dot(xi - xj, np.transpose(xi - xj)))
			record = "%d %d %f " % (i,j,cos)
			result.write(record)
			result.write('\r\n')
result.close()