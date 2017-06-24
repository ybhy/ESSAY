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
# from sklearn.datasets.samples_generator import make_blobs
#························计算tfidf·························
corpus = []
for line in open('t1000.txt', 'r').readlines():
	corpus.append(line.strip())
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
word = vectorizer.get_feature_names()
weight = tfidf.toarray()
# print weight
print 'Features length: ' + str(len(word))
print "weight" , len(weight)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)             #输出两维
newData = pca.fit_transform(weight)   #载入N维
print newData

#·························MS聚类·························
bandwidth = estimate_bandwidth(newData, quantile=0.15, n_samples=1000)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(newData)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# Plot result
import matplotlib.pyplot as plt
from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D

plt.figure(1)
plt.clf()

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    ax.scatter(newData[my_members, 0], newData[my_members, 1], newData[my_members, 2], marker = 'o', c = col)
    ax.scatter(cluster_center[0], cluster_center[1], cluster_center[2], '^')
    # plt.plot(newData[my_members, 0], newData[my_members, 1], newData[my_members, 2], col + '.')
    # plt.plot(cluster_center[0], cluster_center[1], cluster_center[2], 'o', col + 'o',
             # markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
