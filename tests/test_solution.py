import io
import json
from pathlib import Path
import sys
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


BASE = Path(__file__).resolve().parents[1]
APP = BASE / 'wine_app'
sys.path.insert(0, str(APP))
from prediction import predict_batch, read_wines_csv

namespace = {'np': np, 'BaseEstimator': BaseEstimator, 'ClassifierMixin': ClassifierMixin}
notebook = json.loads((BASE / 'HW3_MMDA.ipynb').read_text())
exec(''.join(notebook['cells'][7]['source']), namespace)
LogReg = namespace['LogReg']


class LogisticRegressionTests(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(40), rng.normal(size=(40, 4))])
        y = rng.integers(0, 2, size=40)
        model = LogReg()
        w = rng.normal(size=5)
        model.w = w.copy()
        gradient = model.calc_gradient(X, y)
        numeric = []
        for i in range(len(w)):
            direction = np.eye(len(w))[i] * 1e-6
            model.w = w + direction
            plus = model.calc_loss(X, y)
            model.w = w - direction
            minus = model.calc_loss(X, y)
            numeric.append((plus - minus) / 2e-6)
        np.testing.assert_allclose(gradient, numeric, atol=1e-7)

    def test_extreme_logits_are_finite(self):
        model = LogReg()
        z = np.array([-1e4, 0, 1e4])
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            np.testing.assert_allclose(model.sigmoid(z), [0, 0.5, 1])
            model.w = np.array([1e4])
            self.assertTrue(np.isfinite(model.calc_loss(np.ones((2, 1)), np.array([0, 1]))))

    def test_sgd_reproducibility_and_full_training_loss_history(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(80, 3))
        y = rng.integers(0, 2, size=80)
        w0 = np.array([0.2, -0.1, 0.3, 0.4])
        original = w0.copy()
        model = LogReg(gd_type='stochastic', w0=w0, max_iter=12, tolerance=0, batch_size=7).fit(X, y)
        replica = LogReg(gd_type='stochastic', w0=w0, max_iter=12, tolerance=0, batch_size=7).fit(X, y)
        np.testing.assert_array_equal(w0, original)
        np.testing.assert_array_equal(model.w, replica.w)
        z = np.column_stack([np.ones(len(X)), X]) @ w0
        self.assertAlmostEqual(model.loss_history[0], np.mean(np.logaddexp(0, z)-y*z))
        self.assertEqual(len(model.loss_history), 12)
        np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)
        np.testing.assert_array_equal(model.predict(X, 0), np.ones(len(X)))
        np.testing.assert_array_equal(model.predict(X, 1), np.zeros(len(X)))

    def test_stopping_and_intercept(self):
        model = LogReg(max_iter=20).fit(np.zeros((10, 2)), np.tile([0, 1], 5))
        self.assertEqual(model.stop_reason_, 'tolerance')
        self.assertEqual(model.n_iter_, 1)
        self.assertEqual(model.w.shape, (3,))


class BatchPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(APP / 'model.joblib')
        cls.data = pd.read_csv(APP / 'example_wines.csv')
        cls.features = list(cls.model.feature_names_in_)

    def test_reordered_columns_and_csv_roundtrip(self):
        data = self.data[self.features[::-1]]
        result = predict_batch(self.model, data, self.features, 0.23635860938278724)
        np.testing.assert_allclose(result.probability_good, self.model.predict_proba(self.data)[:, 1])
        decoded = read_wines_csv(result.to_csv(index=False).encode('utf-8-sig'))
        self.assertEqual(len(decoded), len(data))
        np.testing.assert_array_equal(decoded.prediction, result.prediction)

    def test_invalid_inputs_give_clear_errors(self):
        cases = [self.data.drop(columns='alcohol'), self.data.iloc[:0],
                 self.data.assign(alcohol=np.nan), self.data.assign(alcohol='bad'),
                 self.data.assign(alcohol=np.inf), self.data.assign(is_red=2)]
        for data in cases:
            with self.subTest(columns=list(data.columns)), self.assertRaises(ValueError):
                predict_batch(self.model, data, self.features, 0.5)
        for content in [b'', b'alcohol,alcohol\n1,2\n']:
            with self.assertRaises(ValueError):
                read_wines_csv(content)


if __name__ == '__main__':
    unittest.main()
