from collections import defaultdict

import numpy as np
import pandas as pd
from nltk import word_tokenize

from hybrid_search_engine import nlp_engine
import hybrid_search_engine.utils.text_processing as processing


def build_index_from_df(df: pd.DataFrame, columns, id_column, filtering_columns=[], min_token_len=1):
    df = processing.process_df(df, text_columns=columns, lemmatize=True, remove_stopwords=True, lower=True)

    postings, frequencies = build_postings(df, columns)

    index = convert_postings_to_df(postings, frequencies, columns)
    ids = df[id_column].values
    norms = calculate_norms(index, columns, n_docs=len(ids))
    document_tags = df[filtering_columns].copy(deep=True)

    documents_df = pd.concat([df[id_column], norms, document_tags], axis=1)

    return index, documents_df


def calculate_norms(index, columns, n_docs):
    norms = pd.DataFrame()

    for c in columns:
        document_token_num = defaultdict(int)
        document_idxs = index[c].values
        document_frequencies = index[f"{c} TF"].values

        for documents, frequencies in zip(document_idxs, document_frequencies):
            for d, f in zip(documents, frequencies):
                document_token_num[d] += f

        norms[f"{c} Norm"] = [1 / np.sqrt(document_token_num[i]) if document_token_num[i] > 0 else 0 for i in range(n_docs)]

    return norms


def build_postings(corpus, columns):
    postings = dict()
    frequencies = dict()

    for i, document in corpus.iterrows():
        for column in columns:
            if len(document[column]) > 0:
                unique_tokens = list(sorted(set(document[column])))
                for token in unique_tokens:
                    if token in postings:
                        if column in postings[token]:
                            postings[token][column].append(i)
                            frequencies[token][column].append(document[column].count(token))
                        else:
                            postings[token][column] = [i]
                            frequencies[token][column] = [document[column].count(token)]
                    else:
                        postings[token] = {
                            column: [i]
                        }
                        frequencies[token] = {
                            column: [document[column].count(token)]
                        }

    return postings, frequencies


def convert_postings_to_df(postings, frequencies, columns):
    postings_df = pd.DataFrame({
        "token": [k for k in postings.keys()],
        "token vector": [nlp_engine(k).vector for k in postings.keys()]
    })

    postings_df["token vector"] = postings_df["token vector"].apply(lambda v: v / np.linalg.norm(v))

    for column in columns:
        postings_df[column] = [[] for _ in range(len(postings.keys()))]
        postings_df[f"{column} TF"] = [[] for _ in range(len(postings.keys()))]

    for i, token in enumerate(postings.keys()):
        for column, doc_ids in postings[token].items():
            postings_df.loc[i, column] = np.array(doc_ids)
            postings_df.loc[i, f"{column} TF"] = np.array(frequencies[token][column])

    v_dim = nlp_engine("").vector.shape[0]
    mask = np.sum(postings_df["token vector"].values.tolist(), axis=1)
    postings_df.loc[pd.isna(mask), "token vector"] = [np.zeros(v_dim)]

    return postings_df
