#!/usr/bin/env python
# coding=utf-8  
import time          
import re          
import os  
import sys
import codecs
import shutil
import numpy as np
import matplotlib
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer 
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
#························计算tfidf·························
corpus = []
for line in open('text600.txt', 'r').readlines():
	corpus.append(line.strip())
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
word = vectorizer.get_feature_names()
weight = tfidf.toarray()
print weight
print 'Features length: ' + str(len(word))
# resName = "tfidfms600.txt"
# result = codecs.open(resName, 'w', 'utf-8')
# for i in range(len(weight)):
# 	for j in range(len(word)):
# 		result.write(str(weight[i][j]) + ' ')
# 	result.write('\n')
# result.close()
print "weight" , len(weight)
#·························MS聚类·························

bandwidth = estimate_bandwidth(weight, quantile=0.61, n_samples=600)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(weight)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)