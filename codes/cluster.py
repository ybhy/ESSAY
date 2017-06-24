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
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer 

corpus = []
for line in open('preprocess.txt', 'r').readlines():
	corpus.append(line.strip())
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
word = vectorizer.get_feature_names()
weight = tfidf.toarray()
print 'Features length: ' + str(len(word))
resName = "tfidf.txt"
result = codecs.open(resName, 'w', 'utf-8')
for j in range(len(word)):
	result.write(word[j] + ' ')
result.write('\r\n\r\n')
for i in range(len(weight)):
	for j in range(len(word)):
		result.write(str(weight[i][j]) + ' ')
	result.write('\r\n\r\n')

result.close()
print 'Start Kmeans:'
from sklearn.cluster import KMeans
clf = KMeans(n_clusters=5) 
s = clf.fit(weight)
print s
print(clf.cluster_centers_)
label = []               #存储1000个类标 4个类
print(clf.labels_)
i = 1
while i <= len(clf.labels_):
	print i, clf.labels_[i-1]
	label.append(clf.labels_[i-1])
	i = i + 1
print(clf.inertia_)
	
from sklearn.decomposition import PCA
pca = PCA(n_components=2)             #输出两维
newData = pca.fit_transform(weight)   #载入N维
print newData
#DZ
x1 = []
y1 = []
i = 0
while i < 1789:
	x1.append(newData[i][0])
	y1.append(newData[i][1])
       	i += 1
#HX
x2 = []
y2 = []
i = 1789
while i < 5439:
        	x2.append(newData[i][0])
        	y2.append(newData[i][1])
        	i += 1
#IT
x3 = []
y3 = []
i = 5439
while i < 9013:
        	x3.append(newData[i][0])
        	y3.append(newData[i][1])
        	i += 1
#SX
x4 = []
y4 = []
i = 9013
while i < 12846:
        	x4.append(newData[i][0])
        	y4.append(newData[i][1])
        	i += 1
#TX
x5 = []
y5 = []
i = 12846
while i < 16603:
        	x5.append(newData[i][0])
        	y5.append(newData[i][1])
        	i += 1
plt.plot(x1, y1, 'or')
plt.plot(x2, y2, 'og')
plt.plot(x3, y3, 'ob')
plt.plot(x4, y4, 'ok')
plt.plot(x5, y5, 'oy')
plt.show()
print 'ok'
