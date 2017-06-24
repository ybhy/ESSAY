#!/usr/bin/env python
# coding=utf-8  
from sklearn.feature_extraction.text import HashingVectorizer 
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA
import random
import numpy as np

sim = open("sim_read3000.txt", 'w')
# pre = open("perference2000list.txt", 'w')
# f = open("weight1000.txt", 'w')
corpus = []
for line in open('read3000.txt', 'r').readlines():
	corpus.append(line.strip())
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
print "tfidf"
# print tfidf
print type(tfidf)
word = vectorizer.get_feature_names()
weight = tfidf.toarray()
# print weight
print 'Features length: ' + str(len(word))
print "weight" , len(weight)
print "dimension" , len(weight[0])
# print weight
from sklearn.decomposition import PCA
pca = PCA(n_components=5)             #输出两维
newData = pca.fit_transform(weight)   #载入N维
print newData
print len(newData)
print len(newData[0])
for i in xrange(len(newData)):
	for j in xrange(len(newData)):
		# print newData[i][j]
		# f.write(str(newData[i][j]) + ' ')
		if(i != j):
			xi = newData[i]
			xj = newData[j]
			# print (xi - xj)
			similarity = - np.sqrt(np.dot(xi - xj, np.transpose(xi - xj)))
			record = "%04d %04d %f " % (i + 1, j + 1, similarity)
			sim.write(record)
			sim.write('\n')
	# preference = "%s" % (str(-0.123456789000000))
	# pre.write(preference+'\n')
	# f.write('\n')
sim.close()
# pre.close()
# f.close()