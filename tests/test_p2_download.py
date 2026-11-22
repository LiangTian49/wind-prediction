"""Synthetic byte-range integrity checks for the P2 downloader."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from download_p2_tiles import fetch_asset
from temporal_folds import project_path


class DownloadTests(unittest.TestCase):
    def run_download(self, status, payload, extra_headers):
        full = b'II*\x00abcd'
        asset = dict(filename='tile.tif', url='https://example.invalid/tile.tif',
                     bytes=len(full), etag='"pinned"')
        with tempfile.TemporaryDirectory(dir=project_path('result')) as temporary:
            folder = Path(temporary)
            (folder/'tile.part').write_bytes(full[:4])
            response = MagicMock()
            response.status_code = status
            response.headers = {'ETag': '"pinned"', **extra_headers}
            response.iter_content.return_value = iter([payload])
            response.__enter__.return_value = response
            with patch('download_p2_tiles.requests.get', return_value=response) as request:
                saved = fetch_asset(asset, folder)
                self.assertEqual(request.call_args.kwargs['headers']['Range'], 'bytes=4-')
            self.assertEqual((folder/'tile.tif').read_bytes(), full)
            self.assertEqual(saved['sha256'], hashlib.sha256(full).hexdigest())
            with patch('download_p2_tiles.requests.get', side_effect=AssertionError('No redownload')):
                self.assertEqual(fetch_asset(asset, folder)['sha256'], saved['sha256'])

    def test_resume_byte_range(self):
        self.run_download(206, b'abcd', {'Content-Range': 'bytes 4-7/8'})

    def test_server_ignoring_range(self):
        self.run_download(200, b'II*\x00abcd', {})

    def test_changed_remote_version_rejected(self):
        with tempfile.TemporaryDirectory(dir=project_path('result')) as temporary:
            response = MagicMock()
            response.status_code = 200
            response.headers = {'ETag': '"changed"'}
            response.__enter__.return_value = response
            asset = dict(filename='tile.tif', url='https://example.invalid/tile.tif', bytes=8, etag='"pinned"')
            with patch('download_p2_tiles.requests.get', return_value=response):
                with self.assertRaises(ValueError):
                    fetch_asset(asset, Path(temporary))
            self.assertFalse((Path(temporary)/'tile.tif').exists())


if __name__ == '__main__':
    unittest.main()
