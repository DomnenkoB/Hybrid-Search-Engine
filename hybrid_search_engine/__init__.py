import numpy as np
import en_core_sci_lg
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nlp_engine = en_core_sci_lg.load()
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

covid_related_terms = ['COVID', 'Covid', 'covid', 'COVID19', 'nCoViD', 'COViD', 'CoViD', 'CoVID',
                       'CoViD19', 'SARSCoV2', 'ABSTRACTCOVID', 'CoVid', 'covid19']
sars_related_terms = ['SARS', 'SARSCoV', 'hSARS', 'SARSpp', 'rSARS', 'recSARS', 'vSARS',
                      'SARSr', 'sars', 'Sars', 'SARS3a', 'SARS6', 'SARSFP', 'SARSIFP', 'SARSPTM',
                      'SARS2', 'AbstractSARS', 'btSARS', 'SARScoronavirus', 'SARS3', 'wtSARS', 'SSARS', 'BatSARS',
                      'BrSARS', 'ErSARS', 'pSARS']

covid_vector = nlp_engine("coronavirus disease").vector
sars_vector = nlp_engine("SARS").vector

for term in covid_related_terms:
    nlp_engine.vocab.set_vector(term, covid_vector)

zero_vector = np.zeros(len(covid_vector))
for term in sars_related_terms:
    v = nlp_engine(term)
    if v == zero_vector:
        nlp_engine.vocab.set_vector(term, sars_vector)
