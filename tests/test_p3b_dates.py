"""Reject incomplete or misdirected reused/new source combinations before date analysis."""
import json
import unittest
import pandas as pd
from analyze_p3b_dates import validate_summary
from p3b_contract import CONFIG, OUTPUT
from temporal_folds import project_path


class P3bDateChecks(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(project_path(CONFIG).read_text())
        self.summary = pd.read_csv(project_path(OUTPUT)/'combined_results.csv')

    def test_all_sources_and_duplicate_seed_group_rejection(self):
        expected = validate_summary(self.summary, self.config)
        self.assertEqual(len(expected),45)
        with self.assertRaises(ValueError):
            validate_summary(pd.concat([self.summary.iloc[:-1],self.summary.iloc[[0]]]),self.config)
        with self.assertRaises(ValueError):
            validate_summary(self.summary.iloc[:-1],self.config)

    def test_wrong_source_or_origin_rejected(self):
        wrong = self.summary.copy()
        wrong.loc[0,'source_run_dir'] = str(project_path(OUTPUT)/'wrong_run')
        with self.assertRaises(ValueError):
            validate_summary(wrong,self.config)
        wrong = self.summary.copy()
        wrong.loc[0,'source_stage'] = 'P3_reused'
        with self.assertRaises(ValueError):
            validate_summary(wrong,self.config)
