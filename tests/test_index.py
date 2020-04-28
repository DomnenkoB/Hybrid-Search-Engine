from testtools import TestCase
import pandas as pd
import numpy as np

from hybrid_search_engine import index


class TestIndex(TestCase):
    def test_build_index(self):
        df = pd.DataFrame({
            "Id": [1, 2, 3],
            "Title": ["Cat", "Dog", "Cat"],
            "Data": ["Cat is a mammal", "Mammal is an animal", "Feline"],
            "Tag": ["Animal", "Animal", None]
        })
        filtering_columns = ["Tag"]
        columns = ["Title", "Data"]
        id_column = "Id"

        index_df, documents_df = index.build_index_from_df(df, columns=columns, id_column=id_column,
                                                           filtering_columns=filtering_columns)
        expected_index = pd.DataFrame({
            "token": ["cat", "dog", "mammal", "animal", "feline"],
            "Title": [[0, 2], [1], [], [], []],
            "Data": [[0], [], [0, 1], [1], [2]],
            "Title TF": [[1, 1], [1], [], [], []],
            "Data TF": [[1], [], [1, 1], [1], [1]]
        })
        cols = ["Title", "Data", "Title TF", "Data TF"]

        print(index_df[["token"] + cols])

        for c in cols:
            expected_index[c] = expected_index[c].apply(np.array)

        self.assertEqual(expected_index["token"].values.tolist(), index_df["token"].values.tolist())

        for c in ["Title", "Data", "Title TF", "Data TF"]:
            expected_values = expected_index[c].values.tolist()
            actual_values = index_df[c].values.tolist()

            for v1, v2 in zip(expected_values, actual_values):
                print(v1)
                print(v2)
                self.assertTrue((v1 == v2).all())

        expected_norm_values = np.array([
            [1., 1 / np.sqrt(2)],
            [1., 1 / np.sqrt(2)],
            [1., 1.]
        ])

        expected_ids = np.array([1, 2, 3])

        self.assertTrue((expected_norm_values == documents_df[[f"{c} Norm" for c in columns]].values).all())
        self.assertTrue((expected_ids == documents_df[id_column].values).all())
