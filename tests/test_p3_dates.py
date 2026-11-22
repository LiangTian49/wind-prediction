import unittest
import numpy as np
import pandas as pd
from analyze_p3_dates import gain_distribution


class DateGainTests(unittest.TestCase):
    def test_pooled_losses_and_identity(self):
        ref = pd.DataFrame({'squared': [4., 16.], 'absolute': [2., 4.], 'count': [1, 2]})
        model = ref.copy()
        model['squared'] *= .25
        model['absolute'] *= .5
        weights = np.array([[1, 1], [2, 0], [0, 2]])
        for metric, column in [('RMSE', 'squared'), ('MAE', 'absolute')]:
            point, draws = gain_distribution(model, ref, column, metric, weights)
            self.assertAlmostEqual(point, 50.)
            np.testing.assert_allclose(draws, 50.)
            point, draws = gain_distribution(ref, ref, column, metric, weights)
            self.assertEqual(point, 0.)
            np.testing.assert_array_equal(draws, 0.)

    def test_mismatched_samples_rejected(self):
        a = pd.DataFrame({'squared': [1.], 'count': [1]})
        b = pd.DataFrame({'squared': [1.], 'count': [2]})
        with self.assertRaises(AssertionError):
            gain_distribution(a, b, 'squared', 'RMSE', np.ones((1, 1)))
