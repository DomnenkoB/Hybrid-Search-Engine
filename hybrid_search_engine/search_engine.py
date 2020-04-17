from collections import defaultdict
from functools import reduce

import numpy as np
import pandas as pd
from nltk import word_tokenize
from fuzzywuzzy import fuzz

import hybrid_search_engine
from hybrid_search_engine.utils import text_processing as processing
from hybrid_search_engine.utils.exceptions import SearchEngineException


class SearchEngine():
    def __init__(self, index, documents_df, columns, filtering_columns=[], config=None,
                 nlp_engine=None, syntax_threshold=0.9, semantic_threshold=0.8):
        self.index = index
        self.matrix = np.stack(index["token vector"])
        self.syntax_threshold = syntax_threshold
        self.semantic_threshold = semantic_threshold
        self.document_ids = documents_df[documents_df.columns[0]]
        self.document_idx_mapping = {id_: i for i, id_ in enumerate(self.document_ids)}
        self.documents_norm = documents_df[[f"{c} Norm" for c in columns]]
        self.document_tags = documents_df[filtering_columns]
        self.default_columns = columns
        self.filtering_columns = filtering_columns
        self.doc2token_mapping = self.__create_doc2token_mapping()

        self.lower = True
        self.dynamic_idf_reweighting = False
        self.use_TF = True
        self.use_IDF = True
        self.normalize_query = True
        self.syntax_weight = 0.5
        self.semantic_weight = 0.5
        self.dynamic_idf_reweighting = False

        default_weight = 1 / len(columns)
        self.column_weights = {c: default_weight for c in columns}

        if config is not None:
            self.update_config(config)

        if nlp_engine is None:
            self.nlp_engine = hybrid_search_engine.nlp_engine
        else:
            self.nlp_engine = nlp_engine

    def __create_doc2token_mapping(self):
        doc2token_dictionary_mapping = defaultdict(list)

        for column in self.default_columns:
            document_ids = self.index[column].values
            for i, doc_ids in enumerate(document_ids):
                for doc_id in doc_ids:
                    doc2token_dictionary_mapping[doc_id].append(i)

        for k in doc2token_dictionary_mapping.keys():
            doc2token_dictionary_mapping[k] = list(sorted(set(doc2token_dictionary_mapping[k])))

        doc2token_mapping = pd.DataFrame({
            "document_id": [k for k in doc2token_dictionary_mapping.keys()],
            "token_ids": [np.array(v) for k, v in doc2token_dictionary_mapping.items()]
        })
        doc2token_mapping["document_id"] = self.document_ids[doc2token_mapping["document_id"]]
        doc2token_mapping.set_index(keys="document_id", inplace=True)

        return doc2token_mapping

    def __filter_token_by_doc_ids(self, doc_ids):
        token_ids = self.doc2token_mapping.loc[doc_ids, "token_ids"].values
        token_ids = np.concatenate(token_ids)
        token_ids = np.unique(token_ids)

        return np.sort(token_ids)

    def update_config(self, config):
        if "dynamic_idf_reweighting" in config:
            self.dynamic_idf_reweighting = config["dynamic_idf_reweighting"]
        else:
            self.dynamic_idf_reweighting = False
        if "use_TF" in config:
            self.use_TF = config["use_TF"]
        else:
            self.use_TF = True
        if "use_IDF" in config:
            self.use_IDF = config["use_IDF"]
        else:
            self.use_IDF = True
        if "normalize_query" in config:
            self.normalize_query = config["normalize_query"]
        else:
            self.normalize_query = True

        if "similarity_weight" in config and config["similarity_weight"] is not None:
            for weight in ["syntax_weight", "semantic_weight"]:
                if config["similarity_weight"][weight] < 0:
                    raise SearchEngineException(f"{weight} similarity must be greater than 0")

            self.syntax_weight = config["similarity_weight"]["syntax_weight"]
            self.semantic_weight = config["similarity_weight"]["semantic_weight"]

        if "column_weights" in config and config["column_weights"] is not None:
            for c, weight in config["column_weights"].items():
                if weight < 0:
                    raise SearchEngineException(f"{c} weight must be greater than 0")
            self.column_weights = config["column_weights"]

        if "lower" in config:
            self.lower = config["lower"]

    def find(self, query, doc_ids=[], columns=[], filtering_options={}):
        processed_query = processing.process_string(query, lower=self.lower)
        query_tokens = word_tokenize(processed_query)
        if len(query_tokens) == 0:
            return f"Unable to process query. Query '{query}' has been reduced to empty string by text processing"
        if len(columns) == 0:
            columns = self.default_columns
        if len(doc_ids) > 0:
            token_ids = self.__filter_token_by_doc_ids(doc_ids)
        else:
            token_ids = self.index.index.values

        v = [self.nlp_engine(t).vector for t in query_tokens]
        v = np.array([c / np.linalg.norm(c) for c in v])
        v = np.nan_to_num(v)
        syntax_scores = self.index.loc[token_ids]["token"].apply(syntax_similarity, args=(processed_query,))
        semantic_scores = np.matmul(self.matrix[token_ids], v.T)
        semantic_scores = np.max(semantic_scores, axis=1)
        semantic_scores = (semantic_scores + 1) / 2

        data_cols = [c for c in columns] + [f"{c} TF" for c in columns]
        data = self.index.loc[token_ids][(semantic_scores > self.semantic_threshold) |
                                         (syntax_scores > self.syntax_threshold)][data_cols]

        n_doc = len(self.documents_norm)
        filtering_mask = ((semantic_scores > self.semantic_threshold) | (syntax_scores > self.syntax_threshold))

        relevant_doc_ids = []
        syntax_similarities = []
        semantic_similarities = []
        tfs = []
        idfs = []
        norms = []
        token_ix = []

        for col in columns:
            column_weight = self.column_weights[col]
            ids = data[f"{col}"].values.tolist()
            term_frequencies = data[f"{col} TF"].values.tolist()

            for i, (s1, s2, tf) in enumerate(zip(semantic_scores[filtering_mask],
                                                 syntax_scores[filtering_mask],
                                                 term_frequencies)):
                if len(tf) > 0:
                    semantic_similarities += [column_weight * s1] * len(tf)
                    syntax_similarities += [column_weight * s2] * len(tf)
                    tfs += tf.tolist()
                    idfs += [1 + np.log(n_doc / (1 + len(tf)))] * len(tf)
                    norms += self.documents_norm.loc[ids[i], f"{col} Norm"].values.tolist()

                    relevant_doc_ids += ids[i].tolist()
                    token_ix += [i] * len(tf)

        document_similarities = pd.DataFrame({
            "token_id": token_ix,
            "document_id": relevant_doc_ids,
            "semantic_similarity": semantic_similarities,
            "syntax_similarity": syntax_similarities,
            "tf": tfs,
            "idf": idfs,
            "norm": norms
        })

        document_similarities["semantic_similarity"] = self.semantic_weight * document_similarities[
            "semantic_similarity"]
        document_similarities["syntax_similarity"] = self.syntax_weight * document_similarities[
            "syntax_similarity"]
        document_similarities.set_index(keys=["document_id", "token_id"], inplace=True, drop=False)
        document_similarities.index.rename(["document_index", "token_index"], inplace=True)

        if not self.use_TF:
            document_similarities[document_similarities["tf"] > 0] = 1

        document_similarities["similarity"] = (document_similarities["syntax_similarity"] +
                                            document_similarities["semantic_similarity"]) * \
                                        document_similarities["tf"]

        doc_ids_idx = [self.document_idx_mapping[i] for i in doc_ids]

        if self.use_IDF:
            if self.dynamic_idf_reweighting and len(doc_ids_idx) > 0:
                document_similarities = document_similarities.loc[doc_ids_idx]

                n_doc = len(document_similarities["document_id"].unique())
                document_similarities["frequency"] = document_similarities[["token_id", "document_id"]] \
                    .groupby("token_id")["document_id"].transform("unique").apply(len)

                document_similarities["idf"] = (1 + np.log(n_doc / (1 + document_similarities["frequency"])))

            document_similarities["similarity"] = document_similarities["similarity"] * document_similarities["idf"]
        if self.normalize_query:
            document_similarities["similarity"] = document_similarities["similarity"] * document_similarities["norm"]

        if len(doc_ids_idx) > 0:
            document_similarities = document_similarities.loc[doc_ids_idx][["document_id", "similarity"]]

        result = document_similarities[["document_id", "similarity"]].groupby("document_id").sum()
        if len(filtering_options) > 0:
            masks = []
            for k, v in filtering_options.items():
                if isinstance(v, str):
                    mask = self.document_tags.loc[result.index, k].apply(contains, args=(v, ))
                elif isinstance(v, list):
                    mask = reduce(lambda x, y: x | y, [self.document_tags.loc[result.index, k].apply(contains, args=(e,)) for e in v])
                masks.append(mask)

            filtering_mask = reduce(lambda x, y: x & y, masks)
            result = result[filtering_mask]

        result["document_id"] = self.document_ids[result.index]
        result.set_index("document_id", inplace=True)
        result.sort_values(by="similarity", inplace=True, ascending=False)

        return result


def contains(row, value):
    if isinstance(row, list):
        return value in row
    if isinstance(row, str):
        return value == row
    return False


def syntax_similarity(x, y):
    return fuzz.ratio(x, y) / 100
