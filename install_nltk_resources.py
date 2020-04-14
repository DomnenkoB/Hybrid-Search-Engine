import nltk


resources = ['stopwords', 'punkt', 'wordnet']
for r in resources:
    nltk.download(r)
