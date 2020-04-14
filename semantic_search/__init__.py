import scispacy
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nlp_engine = spacy.load("en_core_sci_lg")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
