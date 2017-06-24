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
from sklearn.cluster import DBSCAN
from sklearn import metrics
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


#························计算tfidf·························
corpus = []
for line in open('text600.txt', 'r').readlines():
  corpus.append(line.strip())
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
word = vectorizer.get_feature_names()
weight = tfidf.toarray()
print 'Features length: ' + str(len(word))
# resName = "tfidftext.txt"
# result = codecs.open(resName, 'w', 'utf-8')
# for i in range(len(weight)):
#   for j in range(len(word)):
#     result.write(str(weight[i][j]) + ' ')
#     # print weight[i][j]
#   result.write('\r\n')
# result.close()
print "weight" , len(weight)
##############################################################################
# Compute DBSCAN

labels_true = []
for i in xrange(0, 300):
  labels_true.append(0)
for i in xrange(300, 600):
  labels_true.append(1)

# weight = StandardScaler().fit_transform(weight)
print weight
db = DBSCAN(eps=0.1, min_samples=50).fit(weight)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(weight, labels))

##############################################################################
# Plot result
# import matplotlib.pyplot as plt

# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = 'k'

#     class_member_mask = (labels == k)

#     xy = weight[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)

#     xy = weight[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=6)

# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()
