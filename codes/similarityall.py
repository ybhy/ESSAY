from __future__ import division
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

for topics in xrange(2, 11):
	print ''
	print topics
	f = open('myresult.txt','a')
	texts = [line.strip() for line in file('t5000.txt')]
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
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]
	lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=topics)
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
			if(i >= 0 and i < 1200):
				if(line[0] >= 0 and line[0] < 1200):
					precise = precise + 1
			elif(i>= 1200 and i < 2400):
				if(line[0] >= 1200 and line[0] < 2400):
					precise = precise + 1
			elif(i>= 2400 and i < 3600):
				if(line[0] >= 2400 and line[0] < 3600):
					precise = precise + 1
			elif(i>= 3600 and i < 4800):
				if(line[0] >= 3600 and line[0] < 4800):
					precise = precise + 1
			else:
				if(line[0] >= 4800 and line[0] < 6000):
					precise = precise + 1
		precise = precise / length
		preciseAll = preciseAll + precise
	preciseAll = preciseAll / size
	f.write('topic = ' + str(topics) +'\t' + str(preciseAll) +'\n' )
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