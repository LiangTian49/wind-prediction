import unittest
import pandas as pd
from audit_newyear_headers import summarize_year


class HeaderInventoryChecks(unittest.TestCase):
    def test_calendar_and_duplicate_rejection(self):
        for year, count in [(2024,2928),(2025,2920)]:
            times=list(pd.date_range(f'{year}-01-01',f'{year+1}-01-01',freq='3h',inclusive='left'))
            report=summarize_year(times,year)
            self.assertTrue(report['headers_complete'])
            self.assertEqual(report['expected_steps'],count)
            self.assertFalse(summarize_year(times[:-1],year)['headers_complete'])
            self.assertFalse(summarize_year(times+[times[0]],year)['headers_complete'])

    def test_wrong_year_and_off_schedule(self):
        times=list(pd.date_range('2025-01-01','2026-01-01',freq='3h',inclusive='left'))
        times[0]+=pd.Timedelta(hours=1)
        report=summarize_year(times,2025)
        self.assertEqual(report['off_schedule_steps'],1)
        self.assertEqual(report['missing_steps'],1)
        self.assertFalse(report['headers_complete'])
        self.assertEqual(summarize_year(times,2023)['selected_steps'],0)
