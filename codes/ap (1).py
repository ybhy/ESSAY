#!/usr/bin/env python
# coding=utf-8  
import time          
import re          
import os  
import sys
import codecs
import random
import shutil
import numpy as np
from numpy import arange
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
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

#························计算tfidf·························
f = open("weight1000.txt", 'w')
corpus = []
for line in open('t1000.txt', 'r').readlines():
	corpus.append(line.strip())
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
print "tfidf"
# print tfidf
print type(tfidf)
word = vectorizer.get_feature_names()
weight = tfidf.toarray()
print weight
print type(weight)
for i in xrange(len(weight)):
	for j in xrange(len(weight[0])):
		print weight[i][j]
		f.write(str(weight[i][j]) + ' ')
	f.write('\r\n')
f.close()

from sklearn.decomposition import PCA
pca = PCA(n_components=5)             #输出两维
newData = pca.fit_transform(weight)   #载入N维
print newData


# #·························AP聚类···························
# # print type(weight)
# af = AffinityPropagation(preference=-0.6).fit(newData)
# cluster_centers_indices = af.cluster_centers_indices_
# labels = af.labels_
# print type(cluster_centers_indices)
# n_clusters_ = len(cluster_centers_indices)
# print labels
# labels_true = []
# for i in xrange(0, 200):
# 	labels_true.append(0)
# for i in xrange(200, 400):
# 	labels_true.append(1)
# for i in xrange(400, 600):
# 	labels_true.append(2)
# for i in xrange(600,800):
# 	labels_true.append(3)
# for i in xrange(800, 1000):
# 	labels_true.append(4)
# print('Estimated number of clusters: %d' % n_clusters_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(newData, labels, metric='sqeuclidean'))

# import matplotlib.pyplot as plt
# from itertools import cycle
# from mpl_toolkits.mplot3d import Axes3D
# plt.close('all')
# plt.figure(1)
# plt.clf()
# fig = plt.figure(1)
# ax = fig.add_subplot(111, projection='3d')
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     class_members = labels == k
#     cluster_center = newData[cluster_centers_indices[k]]
#     ax.scatter(newData[class_members, 0], newData[class_members, 1], newData[class_members, 2], marker = 'o', c = col)
#     ax.scatter(cluster_center[0], cluster_center[1], cluster_center[2], '^')
#     for x in newData[class_members]:
#         plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()