#encoding:utf-8
from __future__ import division
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from sklearn.decomposition import PCA
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA
import random
import numpy as np
from numpy import float64
# f = open('myresult.txt','a')
texts = [line.strip() for line in file('read1000.txt')]
texts_name = [text.split('\t')[0] for text in texts]
texts_lower = [[word for word in document.lower().split()] for document in texts]
from nltk.tokenize import word_tokenize
texts_tokenized = [[word.lower() for word in word_tokenize(document.decode('utf-8'))] for document in texts]
english_stopwords = stopwords.words('english')
texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in texts_tokenized]
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
texts_filtered = [[word for word in document if not word in english_punctuations] for document in texts_filtered_stopwords]
st = LancasterStemmer()
texts_stemmed = [[st.stem(word) for word in docment] for docment in texts_filtered]
all_stems = sum(texts_stemmed, [])
stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]
from gensim import corpora, models, similarities
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
# tfidf = models.TfidfModel(corpus)
# corpus_tfidf = tfidf[corpus]
# print type(corpus_tfidf)
# print type(tfidf)

# sim = open("sim_read3000.txt", 'w')
# pre = open("perference2000list.txt", 'w')
# f = open("weight1000.txt", 'w')
corpuss = []
for line in open('t1000.txt', 'r').readlines():
	corpuss.append(line.strip())
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpuss))
word = vectorizer.get_feature_names()
weight = tfidf.toarray()
from sklearn.decomposition import PCA
pca = PCA(n_components=5)             #输出两维
newData = pca.fit_transform(weight)   #载入N维

lsi = models.LsiModel(newData, id2word=dictionary, num_topics=45)
index = similarities.MatrixSimilarity(lsi[corpus])
# print texts_name[70]
# ml_text = texts[70]
# ml_bow = dictionary.doc2bow(ml_text)
# ml_lsi = lsi[ml_bow]
# sims = index[ml_lsi]
# sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
# print sort_sims[0:100]
# f1 = open('res1', 'w') 
# f2 = open('p', 'w') 
size = 1000
length = 200
preciseAll = 0
for i in xrange(len(texts)):
	# print texts_name[i]
	ml_text = texts[i]
	ml_bow = dictionary.doc2bow(ml_text)
	ml_lsi = lsi[ml_bow]
	sims = index[ml_lsi]
	sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
	# print sort_sims[0:length]
	recall = 0
	precise = 0
	for line in sort_sims[0:length]:
		# print line[0]
		# line = open('res1', 'w')		
		# f1.write(str(line[0]) + '\t' + str(line[1])+'\n')
		if(i >= 0 and i < 200):
			if(line[0] >= 0 and line[0] < 200):
				precise = precise + 1
		elif(i>= 200 and i < 400):
			if(line[0] >= 200 and line[0] < 400):
				precise = precise + 1
		elif(i>= 400 and i < 600):
			if(line[0] >= 400 and line[0] < 600):
				precise = precise + 1
		elif(i>= 600 and i < 800):
			if(line[0] >= 600 and line[0] < 800):
				precise = precise + 1
		else:
			if(line[0] >= 800 and line[0] < 1000):
				precise = precise + 1
	precise = precise / length
	preciseAll = preciseAll + precise
preciseAll = preciseAll / size
# f.write('topic = ' + str(topics) +'\t' + str(preciseAll) +'\n' )
print preciseAll
	# print precise
	# if(i >= 0 and i < 100):
	# 	precisea = precisea + precise
	# elif(i >= 100 and i < 200):
	# 	preciseb = preciseb + precise
	# else:
	# 	precisec = precisec + precise 
# precisea = precisea / sizea
# preciseb = preciseb / sizeb
# precisec = precisec / sizec
# print precisea
# print preciseb
# print precisec
	# f.write('\n precise :' +  str(precise) + '\n') 
# f1.close()
# f2.close()
# f.write((tfidf)
# f.close()
print 'ok'