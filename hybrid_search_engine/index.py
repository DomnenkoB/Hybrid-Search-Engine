from collections import defaultdict
import logging
import time

import numpy as np
import pandas as pd
from nltk import word_tokenize
from rank_bm25 import BM25Okapi

from hybrid_search_engine import nlp_engine
import hybrid_search_engine.utils.text_processing as processing


def build_index_from_df(df: pd.DataFrame, columns, id_column, filtering_columns=[], min_token_len=1,
                        lemmatize=True, remove_stopwords=True, lower=True, verbose=False):
    if verbose:
        print(f"{time.ctime()}\t Starting Dataframe text processing")

    df = processing.process_df(df, text_columns=columns, lemmatize=lemmatize,
                               remove_stopwords=remove_stopwords, lower=lower)

    if verbose:
        print(f"{time.ctime()}\t Building postings")
    postings, frequencies = build_postings(df, columns)

    if verbose:
        print(f"{time.ctime()}\t Converting postings to df format")
    index = convert_postings_to_df(postings, frequencies, columns)
    ids = df[id_column].values

    if verbose:
        print(f"{time.ctime()}\t Calculating document norms")
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

    for column in columns:
        documents = corpus[column].values.tolist()

        bm25 = BM25Okapi(documents)

        for i, doc in enumerate(bm25.doc_freqs):
            for token, frequency in doc.items():
                if token in postings:
                    if column in postings[token]:
                        postings[token][column].append(i)
                        frequencies[token][column].append(frequency)
                    else:
                        if column in postings:
                            postings[token][column] = [i]
                            frequencies[token][column] [frequency]
                        else:
                            postings[token][column] = [i]
                            frequencies[token][column] = [frequency]
                else:
                    postings[token] = {column: [i]}
                    frequencies[token] = {column: [frequency]}

                #
                # if token in postings:
                #     if column in postings[token]:
                #         postings[token][column].append(i)
                #     else:
                #         postings[token] = {column: [i]}
                # else:
                #     postings[token] = {column: [i]}
                # if token in frequencies:
                #     if column in frequencies[token]:
                #         frequencies[token][column].append(frequency)
                #     else:
                #         frequencies[token] = {column: [frequency]}
                # else:
                #     frequencies[token] = {
                #         column: [frequency]
                #     }
    #
    # for i, document in corpus.iterrows():
    #     for column in columns:
    #         if len(document[column]) > 0:
    #             unique_tokens = list(sorted(set(document[column])))
    #             for token in unique_tokens:
    #                 if token in postings:
    #                     if column in postings[token]:
    #                         postings[token][column].append(i)
    #                         frequencies[token][column].append(document[column].count(token))
    #                     else:
    #                         postings[token][column] = [i]
    #                         frequencies[token][column] = [document[column].count(token)]
    #                 else:
    #                     postings[token] = {
    #                         column: [i]
    #                     }
    #                     frequencies[token] = {
    #                         column: [document[column].count(token)]
    #                     }

    return postings, frequencies


def convert_postings_to_df(postings, frequencies, columns):
    postings_df = pd.DataFrame({
        "token": [k for k in postings.keys()],
    })

    postings_df["token vector"] = postings_df["token"].apply(lambda t: nlp_engine(t).vector)
    postings_df["token vector"] = postings_df["token vector"].apply(lambda v: v / np.linalg.norm(v))
    #
    # for column in columns:
    #     postings_df[column] = [np.array([]) for _ in range(len(postings.keys()))]
    #     postings_df[f"{column} TF"] = [np.array([]) for _ in range(len(postings.keys()))]

    for column in columns:
        postings_df[column] = postings_df["token"].apply(lambda t: postings[t].get(column, []))
        postings_df[f"{column} TF"] = postings_df["token"].apply(lambda t: frequencies[t].get(column, []))

        postings_df[column] = postings_df[column].apply(lambda x: np.array(x, dtype=np.int32))
        postings_df[f"{column} TF"] = postings_df[f"{column} TF"].apply(lambda x: np.array(x, dtype=np.int32))
    #
    # for i, token in enumerate(postings.keys()):
    #     for column, doc_ids in postings[token].items():
    #         postings_df.loc[i, column] = np.array(doc_ids)
    #         postings_df.loc[i, f"{column} TF"] = np.array(frequencies[token][column])

    v_dim = nlp_engine("").vector.shape[0]
    mask = np.sum(postings_df["token vector"].values.tolist(), axis=1)
    postings_df.loc[pd.isna(mask), "token vector"] = [np.zeros(v_dim)]

    return postings_df
