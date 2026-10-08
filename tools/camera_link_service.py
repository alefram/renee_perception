#!/usr/bin/env python3
"""Camera link service: serves ZED frames to the PC over TCP on request.

Runs on the Jetson next to the ZED 2i and implements the server side of
renee_action_servers/docs/camera_link_protocol.md (version 1). The ZED is
opened ONCE at start-up and kept warm by an idle grab loop (no depth), so a
request only pays for the frames it asks for. Nothing is written to disk.

Target runtime is the Jetson's Python 3.6: standard library + numpy + pyzed
only (no f-string-only features, no dataclasses, no time.time_ns). Note this
differs from the rest of the repo, which needs Python >= 3.10.

    python3 tools/camera_link_service.py                       # ZED, 127.0.0.1:7788
    python3 tools/camera_link_service.py --host 0.0.0.0        # reachable without a tunnel
    python3 tools/camera_link_service.py --source synthetic    # no camera: protocol/network tests

The ZED can only be held by one process: stop anything else that opens it
(e.g. the ROS1 zed_wrapper started by zed_bringup) before starting this.
"""

import argparse
import glob
import json
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import zlib

import numpy as np

PROTOCOL_VERSION = 1
MAX_HEADER_BYTES = 1 << 20
MAX_BLOB_BYTES = 64 << 20

RESOLUTIONS = {  # name -> (width, height)
    'HD2K': (2208, 1242),
    'HD1080': (1920, 1080),
    'HD720': (1280, 720),
    'VGA': (672, 376),
}
DEPTH_MODES = ('PERFORMANCE', 'QUALITY', 'ULTRA', 'NEURAL')


def now_ns():
    # Python 3.6 has no time.time_ns(); a double still resolves ~250 ns at today's epoch.
    return int(time.time() * 1e9)


def log(message):
    print('%s %s' % (time.strftime('%H:%M:%S'), message), flush=True)


# --------------------------------------------------------------------------- wire format

# Best-effort kernel receive timestamps (SO_TIMESTAMPNS, CLOCK_REALTIME) for t1. Known issue:
# pyzed's grab() holds the GIL while it waits for the next frame, so with the camera running the
# connection thread answers up to ~1 frame period late (measured: ping RTT p50 100 ms vs 5 ms
# with the synthetic camera) and a t1 taken in Python is late too, biasing the offset by half the
# delay. The kernel stamp would fix t1, but the Jetson's 4.9 kernel delivers no stamp (empty
# ancillary data, verified), so this falls back to now_ns(). The real fix is moving the grab loop
# into a separate process; until then the PC should use the minimum-RTT ping of several.
SO_TIMESTAMPNS = getattr(socket, 'SO_TIMESTAMPNS', 35)
_TIMESPEC = struct.Struct('ll')


def enable_rx_timestamps(sock):
    try:
        sock.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
        return True
    except OSError:
        return False


def _recv_with_stamp(sock, count):
    """recv() that also returns the kernel receive time in ns (None when unavailable)."""
    data, ancdata, _flags, _addr = sock.recvmsg(count, 128)
    for level, kind, payload in ancdata:
        if level == socket.SOL_SOCKET and kind == SO_TIMESTAMPNS and len(payload) >= _TIMESPEC.size:
            sec, nsec = _TIMESPEC.unpack(payload[:_TIMESPEC.size])
            return data, sec * 1000000000 + nsec
    return data, None


def recv_exact(sock, count, stamp=None):
    """Read exactly `count` bytes. If `stamp` is a list, append the kernel arrival time of the
    first chunk (or None)."""
    chunks = []
    first = stamp is not None
    while count > 0:
        if first:
            chunk, arrival = _recv_with_stamp(sock, min(count, 1 << 20))
            stamp.append(arrival)
            first = False
        else:
            chunk = sock.recv(min(count, 1 << 20))
        if not chunk:
            raise ConnectionError('peer closed the connection')
        chunks.append(chunk)
        count -= len(chunk)
    return b''.join(chunks)


def read_message_ts(sock):
    """(header, blobs, arrival_ns): arrival is when the first byte reached the kernel, or the
    time the header was fully read when the kernel gave no timestamp."""
    stamp = []
    (header_len,) = struct.unpack('>I', recv_exact(sock, 4, stamp))
    if header_len > MAX_HEADER_BYTES:
        raise ConnectionError('header too large (%d bytes)' % header_len)
    header = json.loads(recv_exact(sock, header_len).decode('utf-8'))
    arrival = stamp[0] if stamp and stamp[0] is not None else now_ns()
    sizes = header.get('blob_sizes', [])
    if sum(sizes) > MAX_BLOB_BYTES:
        raise ConnectionError('blobs too large')
    blobs = [recv_exact(sock, size) for size in sizes]
    return header, blobs, arrival


def read_message(sock):
    header, blobs, _arrival = read_message_ts(sock)
    return header, blobs


def send_message(sock, header, blobs=()):
    header = dict(header)
    header['version'] = PROTOCOL_VERSION
    header['blob_sizes'] = [len(blob) for blob in blobs]
    payload = json.dumps(header).encode('utf-8')
    sock.sendall(struct.pack('>I', len(payload)) + payload)
    for blob in blobs:
        sock.sendall(blob)


# --------------------------------------------------------------------------- pixel helpers

def depth_to_mm(depth_m, max_m):
    """float depth in metres -> little-endian uint16 millimetres; 0 = invalid.

    NaN, +-inf, non-positive and beyond max_m (or beyond what uint16 can hold)
    all become 0.
    """
    limit_mm = min(float(max_m) * 1000.0, 65535.0)
    mm = np.asarray(depth_m, dtype=np.float32) * np.float32(1000.0)
    with np.errstate(invalid='ignore'):
        invalid = ~np.isfinite(mm) | (mm <= 0.0) | (mm > limit_mm)
    mm[invalid] = 0.0
    return np.rint(mm).astype('<u2')


def make_camera_info(width, height, fx, fy, cx, cy):
    """Rectified left camera: plumb_bob with zero distortion (VIEW.LEFT is rectified)."""
    return {
        'width': int(width), 'height': int(height), 'distortion_model': 'plumb_bob',
        'D': [0.0, 0.0, 0.0, 0.0, 0.0],
        'K': [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        'P': [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    }


def usb_diagnosis():
    """One-line hint about why the ZED cannot be opened, from sysfs (no sudo, no pyzed)."""
    import glob
    found = {}
    for dev in glob.glob('/sys/bus/usb/devices/*'):
        try:
            with open(dev + '/idVendor') as f:
                if f.read().strip() != '2b03':
                    continue
            with open(dev + '/idProduct') as f:
                pid = f.read().strip()
            with open(dev + '/speed') as f:
                found[pid] = f.read().strip()
        except (IOError, OSError):
            continue
    if not glob.glob('/dev/video*') or 'f880' not in found:
        return 'no ZED video interface (2b03:f880) on USB: check the cable and power, then re-plug the ZED'
    if found['f880'] in ('480', '12'):
        return ('the ZED video interface is on USB 2.0 (%sM); the ZED 2i needs a USB 3 link (5000M): '
                're-plug the cable or try another USB 3 port' % found['f880'])
    return 'ZED present at %sM; another process may be holding it' % found['f880']


class CameraError(Exception):
    pass


class Frame(object):
    __slots__ = ('ts_ns', 'bgr', 'depth_mm')

    def __init__(self, ts_ns, bgr, depth_mm):
        self.ts_ns = ts_ns
        self.bgr = bgr
        self.depth_mm = depth_mm


# --------------------------------------------------------------------------- camera backends

class ZedBackend(object):
    """The real ZED 2i through pyzed (SDK 3.x/4.x). Runs only on the grab thread."""

    def __init__(self, args):
        self.args = args
        self.cam = None
        self.sl = None
        self.camera_info = None
        self._deltas = []
        self.clock_delta_ns = 0

    def open(self):
        import pyzed.sl as sl
        self.sl = sl
        if not glob.glob('/dev/video*'):
            # pyzed has been seen to segfault when the video interface is missing: do not call it.
            raise CameraError('ZED not available: %s' % usb_diagnosis())
        cam = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, self.args.resolution)
        init.camera_fps = self.args.fps
        init.coordinate_units = sl.UNIT.METER
        init.depth_mode = getattr(sl.DEPTH_MODE, self.args.depth_mode)
        init.depth_minimum_distance = self.args.depth_min
        init.depth_maximum_distance = self.args.depth_max
        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            cam.close()
            raise CameraError('ZED open failed: %s (%s)' % (status, usb_diagnosis()))
        self.cam = cam
        info = cam.get_camera_information()
        config = getattr(info, 'camera_configuration', info)
        left = config.calibration_parameters.left_cam
        # SDK 3.x calls it camera_resolution, 4.x resolution; left_cam.image_size is the size
        # the intrinsics belong to, so use it when present.
        res = getattr(config, 'camera_resolution', None) or getattr(config, 'resolution')
        width, height = int(res.width), int(res.height)
        size = getattr(left, 'image_size', None)
        if size is not None and int(size.width) > 0:
            width, height = int(size.width), int(size.height)
        self.camera_info = make_camera_info(width, height, left.fx, left.fy, left.cx, left.cy)
        self.runtime = sl.RuntimeParameters()
        self.image_mat = sl.Mat()
        self.depth_mat = sl.Mat()
        self._deltas = []
        log('ZED opened: serial %s, %dx%d @ %d fps, depth %s' % (
            info.serial_number, width, height,
            self.args.fps, self.args.depth_mode))

    def close(self):
        if self.cam is not None:
            try:
                self.cam.close()
            finally:
                self.cam = None

    def grab(self, want_depth):
        """One grab; returns the image capture time in the Jetson wall clock (ns) or None."""
        sl = self.sl
        self.runtime.enable_depth = bool(want_depth)
        if self.cam.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return None
        image_ts = int(self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
        # The SDK's own clock should equal the wall clock; track the difference
        # (median of recent samples) so the stamp is always in the Jetson wall clock.
        current = int(self.cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds())
        self._deltas.append(now_ns() - current)
        del self._deltas[:-31]
        self.clock_delta_ns = int(sorted(self._deltas)[len(self._deltas) // 2])
        return image_ts + self.clock_delta_ns

    def retrieve(self, want_depth):
        sl = self.sl
        self.cam.retrieve_image(self.image_mat, sl.VIEW.LEFT)
        bgr = np.ascontiguousarray(self.image_mat.get_data()[:, :, :3])  # BGRA view -> BGR copy
        depth_mm = None
        if want_depth:
            self.cam.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)
            depth_mm = depth_to_mm(self.depth_mat.get_data(), self.args.depth_max)
        return bgr, depth_mm


class SyntheticBackend(object):
    """No camera: moving gradient + a depth ramp, paced at --fps. For protocol/network tests."""

    def __init__(self, args):
        self.args = args
        self.width, self.height = RESOLUTIONS[args.resolution]
        self.camera_info = None
        self.clock_delta_ns = 0
        self._next = 0.0
        self._index = 0

    def open(self):
        w, h = self.width, self.height
        rng = np.random.RandomState(0)
        x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
        y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
        noise = rng.randint(0, 8, size=(h, w, 3)).astype(np.float32)
        base = np.empty((h, w, 3), dtype=np.float32)
        base[:, :, 0] = x
        base[:, :, 1] = y
        base[:, :, 2] = 0.5 * (x + y)
        self._rgb = np.clip(base + noise, 0, 255).astype(np.uint8)
        depth = 0.6 + 3.0 * (x / 255.0) + 0.5 * (y / 255.0) + 0.003 * rng.randn(h, w)
        depth[(np.arange(h)[:, None] // 40 + np.arange(w)[None, :] // 40) % 7 == 0] = np.nan
        self._depth = depth.astype(np.float32)
        fx = 0.8 * w
        self.camera_info = make_camera_info(w, h, fx, fx, w / 2.0, h / 2.0)
        self._next = time.time()
        log('synthetic camera: %dx%d @ %.1f fps' % (w, h, self.args.fps))

    def close(self):
        pass

    def grab(self, want_depth):
        self._next += 1.0 / self.args.fps
        delay = self._next - time.time()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next = time.time()
        self._index += 1
        return now_ns()

    def retrieve(self, want_depth):
        bgr = np.roll(self._rgb, (self._index * 3) % self.width, axis=1)
        return bgr, (depth_to_mm(self._depth, self.args.depth_max) if want_depth else None)


# --------------------------------------------------------------------------- grab thread

class _Job(object):
    def __init__(self, after_ns, want_depth):
        self.after_ns = after_ns
        self.want_depth = want_depth
        self.frame = None
        self.done = False
        self.skipped = 0


class CameraWorker(threading.Thread):
    """Owns the camera. Grabs without depth while idle; serves one frame request at a time.

    A request names the earliest acceptable capture time; frames captured before
    it (already buffered in the SDK) are grabbed and dropped, so a returned frame
    always postdates the request.
    """

    def __init__(self, backend, warmup_sec, reopen_delay=2.0, max_grab_failures=20, idle_period=0.2):
        super(CameraWorker, self).__init__(name='camera-worker')
        self.daemon = True
        self.backend = backend
        self.warmup_sec = warmup_sec
        self.reopen_delay = reopen_delay
        self.max_reopen_delay = 30.0  # every failed ZED open resets its USB: back off, don't hammer
        self.open_failures = 0
        self.max_grab_failures = max_grab_failures
        self.ready = threading.Event()
        self.status = 'starting'
        self._stop_flag = threading.Event()
        self._cond = threading.Condition()
        self._pending = None
        self._request_lock = threading.Lock()
        # pyzed's grab() holds the GIL while it waits for the next frame. Grabbing continuously
        # while idle starved every other thread (ping RTT ~100 ms, zlib/send steps stretched to
        # multiples of the frame period), so idle grabs are sparse and stop altogether while a
        # capture is being encoded and sent; the camera itself stays open and warm.
        self.idle_period = idle_period
        self._sessions = 0

    def begin_session(self):
        with self._cond:
            self._sessions += 1

    def end_session(self):
        with self._cond:
            self._sessions -= 1

    def stop(self):
        self._stop_flag.set()

    def run(self):
        failures = 0
        while not self._stop_flag.is_set():
            if not self.ready.is_set():
                self._open()
                failures = 0
                continue
            with self._cond:
                job = self._pending
                if job is None and self._sessions > 0:
                    self._cond.wait(0.05)  # a capture is being encoded/sent: leave the GIL alone
                    continue
            want_depth = bool(job is not None and job.want_depth)
            ts_ns = self.backend.grab(want_depth)
            if ts_ns is None:
                failures += 1
                if failures >= self.max_grab_failures:
                    self._fail('grab failed %d times in a row' % failures)
                continue
            failures = 0
            if job is None:
                with self._cond:
                    if self._pending is None and not self._stop_flag.is_set():
                        self._cond.wait(self.idle_period)  # woken at once by a new request
                continue
            if job.done:
                continue
            if ts_ns < job.after_ns:
                job.skipped += 1
                continue
            try:
                bgr, depth_mm = self.backend.retrieve(want_depth)
            except Exception as exc:  # camera-level failure while retrieving
                self._fail('retrieve failed: %s' % exc)
                continue
            with self._cond:
                job.frame = Frame(ts_ns, bgr, depth_mm)
                job.done = True
                if self._pending is job:
                    self._pending = None
                self._cond.notify_all()
        self.backend.close()

    def _open(self):
        started = time.time()
        self.status = 'opening'
        try:
            self.backend.open()
            warm_until = time.time() + self.warmup_sec
            while time.time() < warm_until and not self._stop_flag.is_set():
                self.backend.grab(False)  # let auto-exposure settle; frames are discarded
        except Exception as exc:
            self.backend.close()
            self.status = 'error: %s' % exc
            self.open_failures += 1
            delay = min(self.max_reopen_delay,
                        self.reopen_delay * 2 ** max(0, self.open_failures - 3))
            log('camera open failed: %s; retrying in %.0f s' % (exc, delay))
            self._stop_flag.wait(delay)
            return
        self.open_failures = 0
        self.status = 'ready'
        self.ready.set()
        log('camera ready after %.1f s (warm-up included)' % (time.time() - started))

    def _fail(self, reason):
        log('camera failure: %s; reopening' % reason)
        self.ready.clear()
        self.status = 'error: %s' % reason
        self.backend.close()
        with self._cond:
            self._cond.notify_all()

    def get_frame(self, after_ns, want_depth, cancel, timeout):
        """Next frame captured at or after after_ns; None if cancelled."""
        if not self._request_lock.acquire(timeout=timeout):
            raise CameraError('camera busy with another request')
        try:
            if not self.ready.is_set():
                raise CameraError('camera not ready (%s)' % self.status)
            job = _Job(after_ns, want_depth)
            deadline = time.time() + timeout
            with self._cond:
                self._pending = job
                self._cond.notify_all()
                while not job.done:
                    if cancel.is_set():
                        self._pending = None
                        return None
                    if not self.ready.is_set():
                        self._pending = None
                        raise CameraError('camera failed (%s)' % self.status)
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        self._pending = None
                        raise CameraError('timed out waiting for a fresh frame')
                    self._cond.wait(min(0.05, remaining))
            return job.frame
        finally:
            self._request_lock.release()


# --------------------------------------------------------------------------- clock status

class ClockMonitor(threading.Thread):
    """Best-effort answer to "has this clock been set?" for the pong message.

    Uses chrony when installed, else timedatectl, else only checks that the
    clock is not still at its boot default. It is a sanity check: the PC
    measures the real offset itself from every ping.
    """

    MIN_PLAUSIBLE_EPOCH = 1704067200  # 2024-01-01

    def __init__(self, period=10.0):
        super(ClockMonitor, self).__init__(name='clock-monitor')
        self.daemon = True
        self.period = period
        self.clock_ok = time.time() > self.MIN_PLAUSIBLE_EPOCH
        self.chrony_offset_ms = None
        self.source = 'none'

    @staticmethod
    def _run(command):
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    universal_newlines=True, timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout if result.returncode == 0 else None

    def refresh(self):
        synced, offset, source = None, None, 'none'
        out = self._run(['chronyc', 'tracking'])
        if out:
            source = 'chrony'
            match = re.search(r'System time\s*:\s*([0-9.eE+-]+) seconds (fast|slow)', out)
            if match:
                offset = float(match.group(1)) * 1000.0 * (1 if match.group(2) == 'fast' else -1)
            leap = re.search(r'Leap status\s*:\s*(.+)', out)
            synced = bool(leap) and leap.group(1).strip() == 'Normal'
        else:
            out = self._run(['timedatectl'])
            if out:
                source = 'timedatectl'
                match = re.search(r'(?:System clock|NTP) synchronized:\s*(\w+)', out)
                synced = bool(match) and match.group(1) == 'yes'
        plausible = time.time() > self.MIN_PLAUSIBLE_EPOCH
        self.clock_ok = bool(plausible and (synced is None or synced))
        self.chrony_offset_ms = offset
        self.source = source

    def run(self):
        while True:
            self.refresh()
            time.sleep(self.period)


# --------------------------------------------------------------------------- server

class Connection(object):
    def __init__(self, sock, addr, server):
        self.sock = sock
        self.addr = addr
        self.server = server
        self.send_lock = threading.Lock()
        self.cancel = threading.Event()
        self.closed = threading.Event()
        self.capture_thread = None

    def close(self):
        self.cancel.set()
        self.closed.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()

    def send(self, header, blobs=()):
        with self.send_lock:
            send_message(self.sock, header, blobs)

    def serve(self):
        log('client connected: %s:%d' % self.addr[:2])
        try:
            while not self.closed.is_set():
                header, _, t1 = read_message_ts(self.sock)  # t1: kernel arrival time of the request
                self.dispatch(header, t1)
        except (ConnectionError, OSError, ValueError) as exc:
            if not self.closed.is_set():
                log('client %s:%d dropped: %s' % (self.addr[0], self.addr[1], exc))
        finally:
            self.close()
            self.server.forget(self)

    def error(self, message):
        log('error -> client: %s' % message)
        self.send({'type': 'error', 'message': message})

    def dispatch(self, header, t1):
        if header.get('version') != PROTOCOL_VERSION:
            self.error('unsupported protocol version %r' % (header.get('version'),))
            return
        kind = header.get('type')
        if kind == 'ping':
            clock = self.server.clock
            with self.send_lock:
                t2 = now_ns()  # latest possible: after any frame in flight, just before sending
                send_message(self.sock, {
                    'type': 'pong', 't0': header.get('t0'), 't1': t1, 't2': t2,
                    'clock_ok': clock.clock_ok, 'chrony_offset_ms': clock.chrony_offset_ms})
        elif kind == 'capture':
            self.start_capture(header, t1)
        elif kind == 'cancel':
            self.cancel.set()
        else:
            self.error('unknown message type %r' % (kind,))

    def start_capture(self, header, t_recv):
        mode, count = header.get('mode'), header.get('num_frames')
        if mode not in ('rgb', 'rgbd'):
            self.error('mode must be "rgb" or "rgbd"')
            return
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            self.error('num_frames must be an integer >= 1')
            return
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.error('a capture is already running on this connection')
            return
        self.cancel.clear()
        self.capture_thread = threading.Thread(
            target=self.run_capture, args=(mode, count, t_recv), name='capture')
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def run_capture(self, mode, count, t_recv):
        worker = self.server.worker
        want_depth = mode == 'rgbd'
        sent, last_ts = 0, 0
        log('capture: mode=%s frames=%d' % (mode, count))
        worker.begin_session()
        try:
            for index in range(count):
                started = time.time()
                frame = worker.get_frame(max(t_recv, last_ts + 1), want_depth, self.cancel,
                                         self.server.frame_timeout)
                if frame is None:
                    break
                grabbed = time.time()
                rgb_blob = zlib.compress(frame.bgr, 1)
                blobs = [rgb_blob]
                height, width = frame.bgr.shape[:2]
                header = {
                    'type': 'frame', 'index': index, 'capture_ts_ns': frame.ts_ns,
                    'rgb': {'width': width, 'height': height, 'encoding': 'bgr8', 'codec': 'zlib'},
                    'camera_info': worker.backend.camera_info,
                }
                if want_depth:
                    blobs.append(zlib.compress(frame.depth_mm, 1))
                    header['depth'] = {'width': width, 'height': height,
                                       'encoding': '16UC1', 'codec': 'zlib'}
                encoded = time.time()
                self.send(header, blobs)
                sent += 1
                last_ts = frame.ts_ns
                log('  frame %d: wait %.0f ms, encode %.0f ms, send %.0f ms, %d bytes' % (
                    index, (grabbed - started) * 1e3, (encoded - grabbed) * 1e3,
                    (time.time() - encoded) * 1e3, sum(len(b) for b in blobs)))
            done = {'type': 'done', 'n_frames': sent}
            if self.cancel.is_set() and sent < count:
                done['cancelled'] = True
            self.send(done)
        except CameraError as exc:
            self.error(str(exc))
        except (ConnectionError, OSError):
            self.closed.set()
        finally:
            worker.end_session()


class CameraLinkServer(object):
    def __init__(self, worker, clock, host, port, frame_timeout):
        self.worker = worker
        self.clock = clock
        self.frame_timeout = frame_timeout
        self.active = None
        self._lock = threading.Lock()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]

    def forget(self, conn):
        with self._lock:
            if self.active is conn:
                self.active = None

    def serve_forever(self):
        while True:
            try:
                sock, addr = self.sock.accept()
            except OSError:
                return
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            enable_rx_timestamps(sock)
            conn = Connection(sock, addr, self)
            with self._lock:
                previous, self.active = self.active, conn
            if previous is not None:
                # One client at a time; a reconnecting PC supersedes a stale (half-open) connection.
                log('new client replaces the previous connection')
                previous.close()
            threading.Thread(target=conn.serve, name='client', daemon=True).start()

    def close(self):
        self.sock.close()
        with self._lock:
            active = self.active
        if active is not None:
            active.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--host', default='127.0.0.1',
                        help='bind address; default only lets the SSH tunnel/jump reach it')
    parser.add_argument('--port', type=int, default=7788)
    parser.add_argument('--source', choices=['zed', 'synthetic'], default='zed')
    parser.add_argument('--resolution', choices=sorted(RESOLUTIONS), default='HD720')
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--depth-mode', choices=DEPTH_MODES, default='NEURAL')
    parser.add_argument('--depth-min', type=float, default=0.3, help='metres')
    parser.add_argument('--depth-max', type=float, default=20.0, help='metres')
    parser.add_argument('--idle-period', type=float, default=0.2,
                        help='seconds between idle grabs (keeps the camera warm without hogging the GIL)')
    parser.add_argument('--warmup-sec', type=float, default=2.0,
                        help='grab-and-discard time after opening so auto-exposure settles')
    parser.add_argument('--frame-timeout', type=float, default=5.0,
                        help='seconds to wait for one fresh frame before failing the capture')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    backend = ZedBackend(args) if args.source == 'zed' else SyntheticBackend(args)
    worker = CameraWorker(backend, args.warmup_sec, idle_period=args.idle_period)
    clock = ClockMonitor()
    clock.refresh()
    clock.start()
    worker.start()
    server = CameraLinkServer(worker, clock, args.host, args.port, args.frame_timeout)
    log('camera link service on %s:%d (source=%s, clock source=%s, clock_ok=%s)' % (
        args.host, server.port, args.source, clock.source, clock.clock_ok))

    def on_term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        log('shutting down')
        server.close()
        worker.stop()
        worker.join(timeout=5)


if __name__ == '__main__':
    sys.exit(main())
