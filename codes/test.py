from gensim import models,corpora,similarities


corpora_documents = ["mama mele maso maso", "ema ma mama"]
corpora_words = [text.split() for text in corpora_documents]

print corpora_words

dictionary = corpora.Dictionary(corpora_words) 
print dictionary

corpora_documents = [text for text in corpora_documents]
print corpora_documents
corpus = [dictionary.doc2bow(text.split()) for text in corpora_documents]  
# corpora_words = test_data_1.
print corpus
similarity = similarities.Similarity('Similarity-index', corpus, num_features=999999999)
print similarity