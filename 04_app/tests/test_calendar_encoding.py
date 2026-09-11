"""Calendar inputs must retain the category representation fitted by N02."""
import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from load_factor_recommender import LoadFactorRecommender

class CalendarEncodingTests(unittest.TestCase):
    def setUp(self):
        self.model = LoadFactorRecommender.__new__(LoadFactorRecommender)
        self.model.features = ['MONTH_OF_YEAR', 'QUARTER_OF_YEAR', 'COVID_PERIOD', 'AIRCRAFT_TYPE', 'LF_LAG_1']
        self.model.categorical_features = self.model.features[:-1]
        self.model.numerical_features = ['LF_LAG_1']

    def test_float_integer_and_missing_calendar_values(self):
        frame = pd.DataFrame({
            'MONTH_OF_YEAR': [12.0, 1, None],
            'QUARTER_OF_YEAR': [4.0, 1, None],
            'COVID_PERIOD': [0.0, 1, None],
            'AIRCRAFT_TYPE': ['001', '335', None],
            'LF_LAG_1': [0.82, None, 0.4],
        })
        original = frame.copy(deep=True)
        prepared = self.model._prepare_feature_frame(frame)
        for feature, values in {
            'MONTH_OF_YEAR': ['12', '1', '__MISSING__'],
            'QUARTER_OF_YEAR': ['4', '1', '__MISSING__'],
            'COVID_PERIOD': ['0', '1', '__MISSING__'],
            'AIRCRAFT_TYPE': ['001', '335', '__MISSING__'],
        }.items():
            self.assertEqual(prepared[feature].tolist(), values)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(prepared.LF_LAG_1.dtype, 'float64')

    def test_fractional_calendar_values_are_not_silently_rounded(self):
        with self.assertRaises((TypeError, ValueError)):
            self.model._prepare_feature_frame(pd.DataFrame({'MONTH_OF_YEAR': [12.5]}))
