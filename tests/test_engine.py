from testtools import TestCase
import pandas as pd
import numpy as np

from hybrid_search_engine import index
from hybrid_search_engine.search_engine import SearchEngine


class TestEngine(TestCase):
    def test_primitive_request(self):
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

        search_engine = SearchEngine(index=index_df, documents_df=documents_df, columns=columns,
                                     filtering_columns=filtering_columns)
        request = "kitten"
        result = search_engine.find(request)

        expected_result = pd.DataFrame({
            "document_id": [1, 3],
            "similarity": np.array([0.525065, 0.263347])
        })
        expected_result.set_index("document_id", inplace=True)

        self.assertTrue((expected_result.index == result.index).all())
        for expected, actual in zip(expected_result["similarity"], result["similarity"]):
            self.assertTrue(np.isclose(expected, actual))
