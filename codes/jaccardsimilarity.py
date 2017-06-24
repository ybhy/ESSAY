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

# for count,line in enumerate(open(r't1000.txt','rU')):
# 	pass
# count += 1
# count = len (open(r't1000.txt','rU')).readlines
# print count
corpus  = dict()
index = 0
for line in open('t_stem_1000.txt', 'r').readlines():
	temp = line.split('\t')
	temp_title = temp[0]
	temp_abstract = temp[1]
	print temp_abstract
	set_word = set()
	temp_abstract = temp_abstract.split(' ')
	print temp_abstract[0]
	print len(temp_abstract)
	for i in range(len(temp_abstract)):
		set_word.add(temp_abstract[i])
	# set_word = set(temp_abstract.strip(' '))
	print set_word
	corpus[index] = set_word
	index = index + 1
print corpus
size = 1000
length = 200
preciseAll = 0
for i in range(0, index):
	d = dict()
	for j in range(0, index):
		if i != j:
			xi = corpus[i]
			# print len(xi)
			xj = corpus[j]
			# print len(xj)
			union = len(xi | xj)
			intersection = len(xi & xj)
			# print union
			# print intersection
			# print len(union)
			# print len(intersection)
			if union == 0:
				jaccard = 0
			else: 
				jaccard = intersection / union
			# record = "%d %d %f " % (i,j,jaccard)
			# print record
			d[j] = jaccard
	sort_euc = sorted(d.iteritems(), key=lambda d:d[1], reverse = False)
	# print sort_euc
	precise = 0
	# for (k,v) in sort_euc.items():
	for line in sort_euc[0:length]:
		k = line[0]
		# print k
		# if i < 1000:
		# 	if(k >= 0 and k < 1000):
		# 		precise = precise + 1
		# elif i <  2000:
		# 	if(k >= 1000 and k < 2000):
		# 		precise = precise + 1
		# else:
		# 	if(k >= 2000 and k < 3000):
		# 		precise = precise + 1
		print k
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
	print precise
	precise = precise / length
	print precise
	preciseAll = preciseAll + precise
	d.clear()
preciseAll = preciseAll / size
print preciseAll
# # result.close()