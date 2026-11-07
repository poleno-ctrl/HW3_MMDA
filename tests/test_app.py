from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.pipeline import Pipeline
import streamlit as st
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / 'wine_app'
sys.path.insert(0, str(APP))


class AppTests(unittest.TestCase):
    def test_tabs_threshold_and_cached_test_predictions(self):
        st.cache_data.clear()
        st.cache_resource.clear()
        original_predict = Pipeline.predict_proba
        test_calls = []

        def tracked_predict(model, X, *args, **kwargs):
            if len(X) == len(pd.read_csv(APP / 'test.csv')):
                test_calls.append(len(X))
            return original_predict(model, X, *args, **kwargs)

        with patch.object(Pipeline, 'predict_proba', new=tracked_predict):
            app = AppTest.from_file(str(APP / 'app.py'), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.tabs), 4)
            self.assertEqual(len(app.number_input), 11)
            self.assertEqual(len(app.toggle), 1)
            old_recall = app.metric[2].value
            app.slider[0].set_value(0.9).run()
            self.assertEqual(len(app.exception), 0)
            self.assertNotEqual(app.metric[2].value, old_recall)
            app.slider[0].set_value(0.0).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.metric[2].value, '1.000')
        self.assertEqual(test_calls, [1596], 'Test predict_proba must remain cached after slider updates')

    def test_valid_and_invalid_uploaded_csv(self):
        with patch('streamlit.file_uploader', return_value=BytesIO((APP / 'example_wines.csv').read_bytes())):
            app = AppTest.from_file(str(APP / 'app.py'), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            self.assertIn('probability_good', app.dataframe[0].value.columns)
        for content in [b'wrong_column\n1\n', b'', b'alcohol,alcohol\n1,2\n']:
            with self.subTest(content=content), patch('streamlit.file_uploader', return_value=BytesIO(content)):
                app = AppTest.from_file(str(APP / 'app.py'), default_timeout=30).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 1)


if __name__ == '__main__':
    unittest.main()
