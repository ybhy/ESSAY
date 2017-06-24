#coding:utf-8
#使用lsh 判断文档相似性

from __future__ import division 
import sklearn
from gensim import models,corpora,similarities
from sklearn.neighbors import LSHForest
from sklearn.neighbors import NearestNeighbors
import numpy as np 
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
import os

corpus = []
for line in open('t2000.txt', 'r').readlines():
	corpus.append(line.strip())
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
print tfidf.toarray()
   
lshf = LSHForest(min_hash_match=4, n_candidates=50, n_estimators=1000,n_neighbors=400, radius=1.0, radius_cutoff_ratio=0.9,random_state=42)
lshf.fit(tfidf.toarray())  
distances, indices = lshf.kneighbors(tfidf.toarray(), n_neighbors=400)  
print(distances)  
print len(distances)
print len(distances[0])
print(indices)
print len(indices)
print len(indices[0])
size = 2000
length = 400
preciseAll = 0
precise = 0
for x in xrange(len(indices)):
	# print x
	for j in xrange(len(indices[0])):
		i = x
		if(x >= 0 and x < 400):
			if(indices[i][j] >= 0 and indices[i][j] < 400):
				precise = precise + 1
		elif(x>= 400 and x < 800):
			if(indices[i][j] >= 400 and indices[i][j] < 800):
				precise = precise + 1
		elif(x>= 800 and x < 1200):
			if(indices[i][j] >= 800 and indices[i][j] < 1200):
				precise = precise + 1
		elif(x>= 1200 and x < 1600):
			if(indices[i][j] >= 1200 and indices[i][j] < 1600):
				precise = precise + 1
		elif(x>= 1600 and x < 2000):
			if(indices[i][j] >= 1600 and indices[i][j] < 2000):
				precise = precise + 1
	precise = precise / length
	# print 'precise ' + str(precise)
	preciseAll = preciseAll + precise
preciseAll = preciseAll / size
print 'preciseAll '  + str(preciseAll)