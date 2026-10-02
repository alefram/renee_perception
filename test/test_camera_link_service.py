#!/usr/bin/env python3
"""Protocol tests for tools/camera_link_service.py (synthetic camera, no ZED needed).

Standard library + numpy only, so they also run on the Jetson's Python 3.6:

    python3 test/test_camera_link_service.py -v

(Run with unittest rather than pytest: this directory's conftest.py needs
open3d/opencv, which the Jetson does not have.)
"""

import os
import socket
import sys
import threading
import time
import unittest
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import camera_link_service as cls  # noqa: E402


class FakeClock(object):
    clock_ok = True
    chrony_offset_ms = None


class DepthToMmTest(unittest.TestCase):
    def test_invalid_values_become_zero(self):
        depth = np.array([[np.nan, np.inf, -np.inf, -1.0, 0.0, 0.5, 19.99, 20.5, 80.0]], np.float32)
        out = cls.depth_to_mm(depth, 20.0)
        self.assertEqual(out.dtype, np.dtype('<u2'))
        self.assertEqual(out.tolist(), [[0, 0, 0, 0, 0, 500, 19990, 0, 0]])

    def test_clamps_to_uint16_range(self):
        out = cls.depth_to_mm(np.array([[70.0]], np.float32), 100.0)
        self.assertEqual(out.tolist(), [[0]])  # 70000 mm does not fit in uint16

    def test_camera_info_layout(self):
        info = cls.make_camera_info(1280, 720, 700.0, 701.0, 640.5, 360.5)
        self.assertEqual(len(info['K']), 9)
        self.assertEqual(len(info['R']), 9)
        self.assertEqual(len(info['P']), 12)
        self.assertEqual(len(info['D']), 5)
        self.assertEqual(info['K'][0], 700.0)
        self.assertEqual(info['P'][5], 701.0)
        self.assertEqual(info['distortion_model'], 'plumb_bob')


class ServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls_):
        args = cls.parse_args(['--source', 'synthetic', '--resolution', 'VGA', '--fps', '30',
                               '--warmup-sec', '0.1'])
        cls_.backend = cls.SyntheticBackend(args)
        cls_.worker = cls.CameraWorker(cls_.backend, args.warmup_sec)
        cls_.worker.start()
        cls_.server = cls.CameraLinkServer(cls_.worker, FakeClock(), '127.0.0.1', 0, 3.0)
        threading.Thread(target=cls_.server.serve_forever, daemon=True).start()
        deadline = time.time() + 5
        while not cls_.worker.ready.is_set() and time.time() < deadline:
            time.sleep(0.05)
        assert cls_.worker.ready.is_set()

    @classmethod
    def tearDownClass(cls_):
        cls_.server.close()
        cls_.worker.stop()

    def connect(self):
        sock = socket.create_connection(('127.0.0.1', self.server.port), timeout=5)
        self.addCleanup(sock.close)
        return sock

    def collect(self, sock):
        frames = []
        while True:
            header, blobs = cls.read_message(sock)
            if header['type'] == 'frame':
                frames.append((header, blobs))
            else:
                return frames, header

    def test_ping(self):
        sock = self.connect()
        before = cls.now_ns()
        cls.send_message(sock, {'type': 'ping', 't0': 123})
        header, _ = cls.read_message(sock)
        after = cls.now_ns()
        self.assertEqual(header['type'], 'pong')
        self.assertEqual(header['t0'], 123)
        self.assertTrue(before <= header['t1'] <= header['t2'] <= after)
        self.assertTrue(header['clock_ok'])
        self.assertIsNone(header['chrony_offset_ms'])

    def test_rgb_capture(self):
        sock = self.connect()
        sent_at = cls.now_ns()
        cls.send_message(sock, {'type': 'capture', 'mode': 'rgb', 'num_frames': 3, 't0': 1})
        frames, done = self.collect(sock)
        self.assertEqual(done['type'], 'done')
        self.assertEqual(done['n_frames'], 3)
        stamps = [h['capture_ts_ns'] for h, _ in frames]
        self.assertEqual([h['index'] for h, _ in frames], [0, 1, 2])
        self.assertTrue(all(b > a for a, b in zip(stamps, stamps[1:])))
        self.assertGreaterEqual(stamps[0], sent_at)  # captured after the request
        header, blobs = frames[0]
        self.assertNotIn('depth', header)
        self.assertEqual(len(blobs), 1)
        width, height = header['rgb']['width'], header['rgb']['height']
        self.assertEqual(header['rgb']['encoding'], 'bgr8')
        self.assertEqual(len(zlib.decompress(blobs[0])), width * height * 3)
        self.assertEqual(header['camera_info']['width'], width)
        self.assertEqual(header['camera_info']['height'], height)

    def test_rgbd_capture(self):
        sock = self.connect()
        cls.send_message(sock, {'type': 'capture', 'mode': 'rgbd', 'num_frames': 2, 't0': 1})
        frames, done = self.collect(sock)
        self.assertEqual(done['n_frames'], 2)
        header, blobs = frames[0]
        self.assertEqual(len(blobs), 2)
        width, height = header['depth']['width'], header['depth']['height']
        self.assertEqual(header['depth']['encoding'], '16UC1')
        depth = np.frombuffer(zlib.decompress(blobs[1]), dtype='<u2').reshape(height, width)
        self.assertTrue((depth == 0).any())            # invalid pixels are 0
        self.assertTrue(((depth > 0) & (depth < 20000)).any())

    def test_bad_requests_keep_connection(self):
        sock = self.connect()
        for request in ({'type': 'capture', 'mode': 'depth', 'num_frames': 1},
                        {'type': 'capture', 'mode': 'rgb', 'num_frames': 0},
                        {'type': 'nonsense'}):
            cls.send_message(sock, request)
            header, _ = cls.read_message(sock)
            self.assertEqual(header['type'], 'error', request)
        cls.send_message(sock, {'type': 'ping', 't0': 1})
        self.assertEqual(cls.read_message(sock)[0]['type'], 'pong')

    def test_wrong_version(self):
        sock = self.connect()
        import json
        import struct
        payload = json.dumps({'version': 99, 'type': 'ping', 'blob_sizes': []}).encode()
        sock.sendall(struct.pack('>I', len(payload)) + payload)
        self.assertEqual(cls.read_message(sock)[0]['type'], 'error')

    def test_cancel(self):
        sock = self.connect()
        cls.send_message(sock, {'type': 'capture', 'mode': 'rgb', 'num_frames': 500, 't0': 1})
        cls.read_message(sock)  # first frame
        cls.send_message(sock, {'type': 'cancel'})
        frames, done = self.collect(sock)
        self.assertEqual(done['type'], 'done')
        self.assertTrue(done.get('cancelled'))
        self.assertLess(done['n_frames'], 500)

    def test_ping_answered_during_capture(self):
        sock = self.connect()
        cls.send_message(sock, {'type': 'capture', 'mode': 'rgb', 'num_frames': 40, 't0': 1})
        cls.send_message(sock, {'type': 'ping', 't0': 77})
        seen_pong = False
        while True:
            header, _ = cls.read_message(sock)
            if header['type'] == 'pong':
                seen_pong = header['t0'] == 77
            if header['type'] == 'done':
                break
        self.assertTrue(seen_pong)

    def test_new_connection_replaces_old(self):
        first = self.connect()
        cls.send_message(first, {'type': 'ping', 't0': 1})
        cls.read_message(first)
        second = self.connect()
        cls.send_message(second, {'type': 'ping', 't0': 2})
        self.assertEqual(cls.read_message(second)[0]['type'], 'pong')
        first.settimeout(2)
        with self.assertRaises((ConnectionError, OSError)):
            cls.read_message(first)


if __name__ == '__main__':
    unittest.main()
