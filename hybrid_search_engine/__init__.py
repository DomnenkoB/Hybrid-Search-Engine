# import scispacy
# import spacy
import en_core_sci_lg
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nlp_engine = en_core_sci_lg.load()
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
