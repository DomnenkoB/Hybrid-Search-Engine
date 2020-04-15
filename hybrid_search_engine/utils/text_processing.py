import re
from functools import reduce
from string import punctuation


import numpy as np
from nltk import word_tokenize

from hybrid_search_engine import nlp_engine, stop_words, lemmatizer


def prepare_corpus(corpus, min_token_len=1, lemmatize=True):
    documents = [word_tokenize(" ".join(d)) for d in corpus]
    if lemmatize:
        documents = [[lemmatizer.lemmatize(t) for t in d] for d in documents]
    documents = [[t for t in d if len(t) >= min_token_len] for d in documents]

    return documents


def prepare_documents(documents, min_token_len=1, lemmatize=True):
    documents = [word_tokenize(d) for d in documents]
    if lemmatize:
        documents = [[lemmatizer.lemmatize(t) for t in d] for d in documents]
    documents = [[t for t in d if len(t) >= min_token_len] for d in documents]

    return documents


def process_df(df, text_columns, lower=True, lemmatize=True, remove_stopwords=True):
    for col in text_columns:
        df[col] = df[col].apply(str)
        if lower:
            df[col] = df[col].apply(str.lower)
        df[col] = df[col].apply(word_tokenize)

        if lemmatize:
            df[col] = df[col].apply(lambda d: [lemmatizer.lemmatize(t) for t in d])
        if remove_stopwords:
            df[col]  = df[col] .apply(lambda d: [t for t in d if t not in stop_words])
        df[col] = df[col].apply(lambda d: [t for t in d if t not in punctuation])

    return df


def extract_corpus_tokens(corpus, min_token_len=1):
    tokens = reduce((lambda x, y: x + y), corpus, [])
    tokens = [word_tokenize(t) for t in tokens if len(t) > 0]
    tokens = reduce((lambda x, y: x + y), tokens, [])
    tokens = list(filter(lambda t: len(t) >= min_token_len, sorted(set(tokens))))

    return tokens


def process_string(s, process_camel_case=False, split_character=["_", "/"], lower=True,
                   lemmatize=True, remove_stopwords=True, trim_punctuation=True):
    processed_string = str(s)

    for c in split_character:
        processed_string = processed_string.replace(c, " ")
    if process_camel_case:
        processed_string = split_camel_case(processed_string)
    if trim_punctuation:
        processed_string = remove_punctuation(processed_string)
    tokens = word_tokenize(processed_string)
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]

    result = " ".join(tokens)
    if lower:
        result = result.lower()

    return result


def split_camel_case(s):
    return " ".join(re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", s)).split())


def remove_punctuation(s):
    return s.strip(punctuation)
