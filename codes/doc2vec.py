#coding:utf-8
#使用doc2vec 判断文档相似性
from __future__ import division 
from gensim import models,corpora,similarities
import jieba.posseg as pseg
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import os

#创建model
i = 1
corpus = []
corpus_doc = []
doc=[] 

texts = [line.strip() for line in file('t1000.txt')]
corpus = [text.split('\t')[1] for text in texts]
# print corpus

# for line in open('text60.txt', 'r').readlines():
for line in xrange(len(corpus)):
	# corpus.append(line.strip())
	document = TaggedDocument(words=corpus, tags=[i])
	corpus_doc.append(document)
	doc.append(corpus)
	i = i+1

#创建model
model = Doc2Vec(size=1000, min_count=1, iter=5)
model.build_vocab(corpus_doc)
model.train(corpus_doc)
print('#########', model.vector_size)

size = 1000
length = 200
preciseAll = 0
# 训练
for x in xrange(len(corpus)):
	test_doc = corpus[x]
	inferred_vector = model.infer_vector(test_doc)
	sims = model.docvecs.most_similar([inferred_vector], topn=length)
	# print(sims) #sims是一个tuples,(index_of_document, similarity)
	precise = 0
	for j in sims:
		# print j[0]
	 	if(x >= 0 and x < 200):
			if(j[0] >= 0 and j[0] < 200):
				precise = precise + 1
		elif(x>= 200 and x < 400):
			if(j[0] >= 200 and j[0] < 400):
				precise = precise + 1
		elif(x>= 400 and x < 600):
			if(j[0] >= 400 and j[0] < 600):
				precise = precise + 1
		elif(x>= 600 and x < 800):
			if(j[0] >= 600 and j[0] < 800):
				precise = precise + 1
		else:
			if(j[0] >= 800 and j[0] < 1000):
				precise = precise + 1
	precise = precise / length
	preciseAll = preciseAll + precise
preciseAll = preciseAll / size
# f.write('topic = ' + str(topics) +'\t' + str(preciseAll) +'\n' )
print preciseAll