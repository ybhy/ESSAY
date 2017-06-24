#!/usr/bin/env python
# -*- coding: utf-8 -*- 
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

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

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
##############################################################################
from sklearn.decomposition import PCA
pca = PCA(n_components=3)             #输出两维
newData = pca.fit_transform(weight)   #载入N维
print newData
# Generate sample data
labels_true = []
for i in xrange(0, 200):
  labels_true.append(0)
for i in xrange(200, 400):
  labels_true.append(1)
for i in xrange(400, 600):
  labels_true.append(2)
for i in xrange(600, 800):
  labels_true.append(3)
for i in xrange(800, 1000):
  labels_true.append(4)
# for i in xrange(2500, 3000):
#   labels_true.append(5)
# print labels_true
newData = StandardScaler().fit_transform(newData)
##############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.1, min_samples=30).fit(newData)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(newData, labels))
##############################################################################
# Plot result
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0,1,len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -2:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xyz = newData[class_member_mask & core_samples_mask]
    plt.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],'^', markerfacecolor=col,
             markeredgecolor='k', markersize=8)

    xyz = newData[class_member_mask & ~core_samples_mask]
    plt.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=8)

    xyz = newData[class_member_mask & ~core_samples_mask]
    plt.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], '*', markerfacecolor=col,
             markeredgecolor='k', markersize=8)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
