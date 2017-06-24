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
from sklearn.feature_extraction import DictVectorizer
#························计算tfidf·························
corpus = []
for line in open('t1000.txt', 'r').readlines():
    line = line.split('\t')
    corpus.append(line[1].strip())
vectorizer = CountVectorizer()
# vectorizer = DictVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
# vec = vectorizer.fit_transform(corpus)
# weight = vec.toarray()
# print len(vec.toarray()[0])
print tfidf
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
size = 1000
length = 200
preciseAll = 0

for i in xrange(0, len(weight)):
    d = dict()
    print i
    for j in xrange(0, len(weight)):
        if(i != j):
            xi = weight[i]
            xj = weight[j]
            intersection = 0
            union = 0

            for k in xrange(0, len(xi)):
                if xi[k] > 1:
                    print "hello"
                if xi[k] > 0 and xj[k] > 0:
                    intersection = intersection + 1
                if xi[k] > 0 or xj[k] > 0:
                    union = union + 1
            if union == 0:
                jaccard = 0
            else:
                jaccard = intersection / union
            # intersection1 = dot(xi, xj)
            # union1 = sum(xi + xj) - intersection1
            # print intersection, intersection1
            d[j] = jaccard
            record = "%d %d %f " % (i,j,jaccard)
            # print "%d %d %f" % (i, j, jaccard)
    sort_j = sorted(d.iteritems(), key=lambda d:d[1], reverse = False)
    # print sort_euc
    precise = 0
    for line in sort_j[0:length]:
        k = line[0]
        # print k
        if i < 200:
            if(k >= 0 and k < 200):
                precise = precise + 1
            elif i <  400:
                if(k >= 200 and k < 400):
                    precise = precise + 1
            elif i <  600:
                if(k >= 400 and k < 600):
                    precise = precise + 1
            elif i <  800:
                if(k >= 600 and k < 800):
                    precise = precise + 1
            else:
                if(k >= 800 and k < 1000):
                    precise = precise + 1
    precise = precise / length
    preciseAll = preciseAll + precise
    d.clear()
preciseAll = preciseAll / size
print preciseAll