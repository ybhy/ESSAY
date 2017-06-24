from __future__ import division
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
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
print len(weight)
print len(weight[0])
size = 3000
length = 200
preciseAll = 0


# resName = "euc60.txt"
# result = codecs.open(resName, 'w', 'utf-8')
for i in xrange(0, len(weight)):
	d = dict()
	for j in xrange(0, len(weight)):
		if(i != j):
			xi = weight[i]
			xj = weight[j]
			euc = numpy.sqrt(numpy.sum(numpy.square(xi - xj)))  
			d[j] = euc
			record = "%d %d %f " % (i,j,euc)
			# result.write(record)
			# result.write('\r\n')

	sort_euc = sorted(d.iteritems(), key=lambda d:d[1], reverse = False)
	# print sort_euc
	precise = 0
	# for (k,v) in sort_euc.items():
	for line in sort_euc[0:length]:
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
# result.close()