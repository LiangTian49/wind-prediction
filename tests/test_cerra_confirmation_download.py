"""Request/product guard tests only: no CDS requests, installation or real GRIB access."""
import unittest
import json
from unittest.mock import Mock, patch

from download_cerra_confirmation import TASKS, download, monthly_request, validate_header
from confirmation_protocol import GRID_KEYS
from cerra_download_remote import matches, resume_or_submit


class DownloadTests(unittest.TestCase):
    def test_24_month_mapping(self):
        self.assertEqual(len(set(TASKS)), 24)
        self.assertEqual(TASKS[0], (2025, 1))
        self.assertEqual(TASKS[11], (2025, 12))
        self.assertEqual(TASKS[12], (2023, 1))
        self.assertEqual(TASKS[23], (2023, 12))

    def test_request_selects_analysis_not_three_hour_forecast(self):
        request = monthly_request(2025, 1)
        self.assertEqual(request['product_type'], 'analysis')
        self.assertEqual(request['data_type'], ['reanalysis'])
        self.assertNotIn('leadtime_hour', request)
        self.assertNotIn('area', request)
        self.assertNotIn('grid', request)
        self.assertEqual(len(request['time']), 8)
        self.assertEqual(len(request['variable']), 2)

    def test_calendar_has_no_invalid_days(self):
        self.assertEqual(monthly_request(2025, 2)['day'][-1], '28')
        self.assertEqual(monthly_request(2023, 4)['day'][-1], '30')
        with self.assertRaises(ValueError):
            monthly_request(2024, 1)

    def test_product_guards_speed_and_direction(self):
        grid = {key: 1 for key in GRID_KEYS}
        grid['units'] = 'm s**-1'
        speed = dict(grid, paramId=207, dataType='an', typeOfLevel='heightAboveGround',
                     level=10, stepRange='0', stepType='instant')
        validate_header(speed, grid)
        validate_header(dict(speed, paramId=260260, units='Degree true'), grid)
        for changes in [dict(dataType='fc'), dict(stepRange='3'), dict(level=100),
                        dict(paramId=260260, units='radians'), dict(paramId=999)]:
            with self.assertRaises(ValueError):
                validate_header(speed | changes, grid)

    def test_download_blocked_on_login_before_network(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'sbatch'):
                download(0)

    def test_remote_matching_accepts_scalar_list_but_rejects_forecast(self):
        request = monthly_request(2025, 3)
        self.assertTrue(matches(request, request | {'product_type': ['analysis']}))
        self.assertFalse(matches(request, request | {'product_type': 'forecast'}))

    def test_submit_saves_id_before_any_wait_or_download(self):
        client, state, remote = Mock(), Mock(), Mock()
        state.exists.return_value = False
        remote.request_id = 'saved-id'
        client.retrieve.return_value = remote
        request = monthly_request(2025, 3)
        with patch('cerra_download_remote.active_jobs', return_value=[]), \
             patch('cerra_download_remote.save_state') as save:
            self.assertIs(resume_or_submit(client, 'reanalysis-cerra-single-levels', request, state), remote)
            self.assertEqual(save.call_args.args[1]['request_id'], 'saved-id')
            remote.download.assert_not_called()

    def test_saved_active_remote_is_resumed_without_submission(self):
        request = monthly_request(2025, 3)
        client, state, remote = Mock(), Mock(), Mock()
        state.exists.return_value = True
        state.read_text.return_value = json.dumps(dict(dataset='reanalysis-cerra-single-levels',
                                                       request=request, request_id='old-id'))
        remote.collection_id = 'reanalysis-cerra-single-levels'
        remote.request, remote.status, remote.request_id = request, 'running', 'old-id'
        client.client.get_remote.return_value = remote
        self.assertIs(resume_or_submit(client, remote.collection_id, request, state), remote)
        client.retrieve.assert_not_called()

    def test_other_active_month_blocks_new_submission(self):
        client, state, remote = Mock(), Mock(), Mock()
        state.exists.return_value = False
        remote.request = monthly_request(2025, 4)
        with patch('cerra_download_remote.active_jobs', return_value=[remote]):
            with self.assertRaisesRegex(ValueError, 'Other CERRA requests'):
                resume_or_submit(client, 'reanalysis-cerra-single-levels', monthly_request(2025, 3), state)
        client.retrieve.assert_not_called()


if __name__ == '__main__':
    unittest.main()
