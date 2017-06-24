from __future__ import division
import sys

texts = [line.strip() for line in file('ex300an.txt')]
texts_name = [text.split('\t')[0] for text in texts]
import nltk
texts_lower = [[word for word in document.lower().split()] for document in texts]
from nltk.tokenize import word_tokenize
texts_tokenized = [[word.lower() for word in word_tokenize(document)] for document in texts]
from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')
texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in texts_tokenized]
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
texts_filtered = [[word for word in document if not word in english_punctuations] for document in texts_filtered_stopwords]
from nltk.stem.lancaster import LancasterStemmer
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
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=3)
index = similarities.MatrixSimilarity(lsi[corpus])
for i in xrange(len(texts)):
	print texts_name[i]
	ml_text = texts[i]
	ml_bow = dictionary.doc2bow(ml_text)
	ml_lsi = lsi[ml_bow]
	sims = index[ml_lsi]
	print type(sims)
	sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
	print sort_sims[0:100]
             # recall = 0
	precise = 0
	for line in sort_sims[0:100]:
		# line = open('result', 'w')		
		# f.write('['+str(line[0]) + ':' + str(line[1])+'];')
		if(i >= 0 and i < 100):
			if(line[0] >= 0 and line[0] < 100):
				precise = precise + 1
		elif(i>= 1000 and i < 200):
			if(line[0] >= 100 and line[0] < 200):
				precise = precise + 1
		else:
			if(line[0] >= 200 and line[0] < 300):
				precise = precise + 1

	precise = precise / 100
	# f.write('\n precise :' +  str(precise) + '\n') 
		# sys.stdout = open('result','w')
# f.close( )
print 'ok'