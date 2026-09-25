#!/usr/bin/env python3
"""Detect ArUco in rectified ZED left images; save camera-side pose only.

GUI: S saves all valid detections in the current frame, Q exits.
SSH: --headless, Enter saves, q then Enter exits.
--save-once: no interaction -- waits for a stable, low-error detection and
saves it automatically, then exits. Meant to be driven remotely (e.g. from
capture_aruco_pose.sh over ssh) as part of an automated arm-pose capture.
T_camera_marker maps marker coordinates into the LEFT optical camera frame:
x right, y down, z forward. Translation is in metres. No robot TF is saved.
Single planar marker poses can be ambiguous (a near-head-on view has two
almost-equally-valid solutions); when the installed OpenCV supports
SOLVEPNP_IPPE_SQUARE both planar solutions are computed and the one with
lower reprojection error is kept, instead of trusting whichever one the
plain iterative solver happens to converge to.
"""
import argparse
import json
import os
from pathlib import Path
import select
import sys
import time
from datetime import datetime, timezone

import numpy as np


def candidate_poses(cv2, object_points, points, K, D):
    """Return a list of (rvec, tvec) candidate solutions for a planar 4-point target."""
    if hasattr(cv2, 'solvePnPGeneric') and hasattr(cv2, 'SOLVEPNP_IPPE_SQUARE'):
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(object_points, points, K, D,
                                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            return list(zip(rvecs, tvecs))
    ok, rvec, tvec = cv2.solvePnP(object_points, points, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    return [(rvec, tvec)] if ok else []


def detect_frame(cv2, sl, camera, runtime, mat, aruco, detector, dictionary, params,
                  object_points, K, D, args):
    """Grab one frame and run ArUco detection. Returns None on a grab failure."""
    status = camera.grab(runtime)
    if status != sl.ERROR_CODE.SUCCESS:
        return None
    camera.retrieve_image(mat, sl.VIEW.LEFT)
    raw = mat.get_data()
    bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR) if raw.shape[2] == 4 else raw.copy()
    capture_time = datetime.now(timezone.utc).isoformat()
    stamp_ns = int(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = (detector.detectMarkers(gray) if detector is not None
                        else aruco.detectMarkers(gray, dictionary, parameters=params))
    annotated = bgr.copy()
    results = []
    if ids is not None:
        aruco.drawDetectedMarkers(annotated, corners, ids)
        for corner, marker_id in zip(corners, ids.flatten()):
            if args.id is not None and int(marker_id) != args.id:
                continue
            points = np.asarray(corner, dtype=np.float64).reshape(4, 2)
            best = None
            for rvec, tvec in candidate_poses(cv2, object_points, points, K, D):
                if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                    continue
                R, _ = cv2.Rodrigues(rvec)
                if np.any((R.dot(object_points.T) + tvec.reshape(3, 1))[2] <= 0):
                    continue
                projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
                error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - points) ** 2, axis=1))))
                if best is None or error < best[2]:
                    best = (rvec, tvec, error)
            if best is None:
                continue
            rvec, tvec, error = best
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, tvec.flatten()
            results.append({'id': int(marker_id), 'corners_px': points.tolist(),
                            'rvec_rad': rvec.flatten().tolist(),
                            'tvec_m': tvec.flatten().tolist(),
                            'reprojection_rmse_px': error,
                            'T_camera_marker': T.tolist()})
            if hasattr(cv2, 'drawFrameAxes'):
                cv2.drawFrameAxes(annotated, K, D, rvec, tvec, args.size * 0.5)
            else:
                aruco.drawAxis(annotated, K, D, rvec, tvec, args.size * 0.5)
            text = 'ID {} XYZ(m) {:.3f} {:.3f} {:.3f} err {:.2f}px'.format(
                int(marker_id), *tvec.flatten(), error)
            cv2.putText(annotated, text, (10, 30 + 28 * (len(results) - 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    return bgr, annotated, results, capture_time, stamp_ns


def save_sample(cv2, root, bgr, annotated, capture_time, stamp_ns, info, K, D, args, results, flat):
    """Write left/annotated PNGs + pose.json.

    flat=True writes straight into `root` (used by --save-once, whose caller
    already knows the destination path); flat=False creates a timestamped
    subfolder per sample (interactive mode, possibly many samples per run).
    """
    root = Path(root)
    folder = root if flat else root / datetime.now(timezone.utc).strftime('sample_%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True, exist_ok=True)
    for name, pixels in [('left.png', bgr), ('annotated.png', annotated)]:
        if not cv2.imwrite(str(folder / name), pixels):
            raise RuntimeError('Image write failed: ' + str(folder / name))
    record = {'host_capture_time_utc': capture_time,
              'zed_image_timestamp_ns': stamp_ns,
              'camera_serial_number': int(info.serial_number),
              'camera_frame': 'ZED rectified LEFT optical: x right, y down, z forward',
              'transform_convention': 'p_camera = T_camera_marker @ p_marker',
              'marker_frame': 'origin at centre; x right, y up on printed marker, z out of face',
              'marker_size_m': args.size, 'dictionary': args.dictionary,
              'image_width': bgr.shape[1], 'image_height': bgr.shape[0],
              'K': K.tolist(), 'distortion': D.flatten().tolist(),
              'markers': results, 'robot_pose_included': False}
    with (folder / 'pose.json').open('w') as f:
        json.dump(record, f, indent=2)
    return folder


def run_save_once(cv2, sl, camera, runtime, mat, info, aruco, detector, dictionary, params,
                   object_points, K, D, args):
    print('Waiting for a stable detection (timeout {:.0f}s)...'.format(args.timeout), flush=True)
    start = time.monotonic()
    failures = 0
    stable_count = 0
    last_tvec = None
    pending = None
    while True:
        if time.monotonic() - start > args.timeout:
            raise SystemExit('No stable detection within {:.1f}s timeout.'.format(args.timeout))
        frame = detect_frame(cv2, sl, camera, runtime, mat, aruco, detector, dictionary, params,
                              object_points, K, D, args)
        if frame is None:
            failures += 1
            if failures >= 100:
                raise RuntimeError('Repeated camera grab failure.')
            time.sleep(0.02)
            continue
        failures = 0
        bgr, annotated, results, capture_time, stamp_ns = frame
        candidates = [r for r in results if r['reprojection_rmse_px'] <= args.max_reprojection_error]
        if not candidates:
            stable_count = 0
            last_tvec = None
            continue
        best = min(candidates, key=lambda r: r['reprojection_rmse_px'])
        if len(candidates) > 1 and args.id is None:
            print('WARNING: {} markers below error threshold in one frame; using ID {} (lowest error).'.format(
                len(candidates), best['id']), flush=True)
        tvec = np.array(best['tvec_m'])
        if last_tvec is not None and np.max(np.abs(tvec - last_tvec)) <= args.stable_tolerance:
            stable_count += 1
        else:
            stable_count = 1
        last_tvec = tvec
        pending = (bgr, annotated, capture_time, stamp_ns, results)
        print('ID={} tvec_m={} err={:.3f}px stable={}/{}'.format(
            best['id'], np.round(tvec, 4).tolist(), best['reprojection_rmse_px'],
            stable_count, args.stable_frames), flush=True)
        if stable_count >= args.stable_frames:
            break
    bgr, annotated, capture_time, stamp_ns, results = pending
    folder = save_sample(cv2, args.output, bgr, annotated, capture_time, stamp_ns, info, K, D,
                          args, results, flat=True)
    print('SAVED:', folder.resolve(), flush=True)


def run_interactive(cv2, sl, camera, runtime, mat, info, aruco, detector, dictionary, params,
                     object_points, K, D, args, headless):
    print('Enter = save, q + Enter = quit' if headless else 'S = save, Q / ESC = quit')
    last_report = 0.0
    failures = 0
    while True:
        frame = detect_frame(cv2, sl, camera, runtime, mat, aruco, detector, dictionary, params,
                              object_points, K, D, args)
        if frame is None:
            failures += 1
            if failures >= 100:
                raise RuntimeError('Repeated camera grab failure.')
            time.sleep(0.02)
            continue
        failures = 0
        bgr, annotated, results, capture_time, stamp_ns = frame
        if time.monotonic() - last_report >= 1:
            for result in results:
                print('ID={} tvec_m={} reprojection={:.3f}px'.format(
                    result['id'], np.round(result['tvec_m'], 4).tolist(),
                    result['reprojection_rmse_px']), flush=True)
            if not results:
                print('No matching marker. Check dictionary, lighting and full black border.', flush=True)
            last_report = time.monotonic()
        save = False
        if headless:
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                command = sys.stdin.readline()
                if command == '' or command.strip().lower() == 'q':
                    break
                save = command.strip().lower() in ('', 's')
        else:
            cv2.imshow('ZED left - ArUco', annotated)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord('q')):
                break
            save = key == ord('s')
        if save:
            if not results:
                print('Not saved: no valid pose in this frame.')
                continue
            folder = save_sample(cv2, args.output, bgr, annotated, capture_time, stamp_ns, info, K, D,
                                  args, results, flat=False)
            print('Saved:', folder.resolve(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=float, default=0.060, help='Outer black square side in metres')
    parser.add_argument('--dictionary', default='DICT_4X4_50')
    parser.add_argument('--id', type=int, default=None, help='Optional marker ID filter')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output', default='aruco_captures')
    parser.add_argument('--save-once', action='store_true',
                         help='Automatic mode: wait for a stable, low-error detection, save it, and '
                              'exit -- no interactive input. Meant to be driven remotely, e.g. by '
                              'capture_aruco_pose.sh over ssh. Writes directly into --output (no '
                              'per-sample subfolder).')
    parser.add_argument('--timeout', type=float, default=15.0,
                         help='--save-once only: seconds to wait for a stable detection before giving up.')
    parser.add_argument('--max-reprojection-error', type=float, default=0.5,
                         help='--save-once only: max reprojection RMSE (px) to accept a detection.')
    parser.add_argument('--stable-frames', type=int, default=3,
                         help='--save-once only: consecutive good frames required before auto-saving.')
    parser.add_argument('--stable-tolerance', type=float, default=0.003,
                         help='--save-once only: max translation change (metres) between consecutive '
                              'frames to count as stable.')
    args = parser.parse_args()
    if args.size <= 0:
        parser.error('--size must be positive')
    try:
        import cv2
        import pyzed.sl as sl
    except ImportError as exc:
        raise SystemExit('Missing dependency in this Python environment: {}'.format(exc))
    if not hasattr(cv2, 'aruco'):
        raise SystemExit('This OpenCV has no aruco module. An OpenCV contrib build is required.')
    aruco = cv2.aruco
    if not args.dictionary.startswith('DICT_') or not hasattr(aruco, args.dictionary):
        raise SystemExit('Unknown ArUco dictionary: ' + args.dictionary)
    dictionary_id = getattr(aruco, args.dictionary)
    dictionary = (aruco.getPredefinedDictionary(dictionary_id)
                  if hasattr(aruco, 'getPredefinedDictionary') else aruco.Dictionary_get(dictionary_id))
    params = (aruco.DetectorParameters() if hasattr(aruco, 'DetectorParameters')
              else aruco.DetectorParameters_create())
    if hasattr(aruco, 'CORNER_REFINE_SUBPIX'):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = aruco.ArucoDetector(dictionary, params) if hasattr(aruco, 'ArucoDetector') else None
    headless = args.headless or args.save_once or not os.environ.get('DISPLAY')
    half = args.size / 2.0
    # ArUco order: top-left, top-right, bottom-right, bottom-left.
    object_points = np.array([[-half, half, 0], [half, half, 0],
                              [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
    camera = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 15
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        camera.close()
        raise SystemExit('ZED open failed: {}. Close other camera programs and check USB.'.format(status))
    try:
        info = camera.get_camera_information()
        config = getattr(info, 'camera_configuration', info)
        left = config.calibration_parameters.left_cam
        K = np.array([[left.fx, 0, left.cx], [0, left.fy, left.cy], [0, 0, 1]], dtype=np.float64)
        # VIEW.LEFT is rectified: use rectified intrinsics and zero distortion.
        D = np.zeros((5, 1), dtype=np.float64)
        mat = sl.Mat()
        runtime = sl.RuntimeParameters()
        print('OpenCV:', cv2.__version__)
        print('Dictionary:', args.dictionary, '| Side (m):', args.size)
        print('Rectified left intrinsics:\n', K)
        print('Axes: x right, y down, z forward. Pose translation: metres.')
        if args.save_once:
            run_save_once(cv2, sl, camera, runtime, mat, info, aruco, detector, dictionary, params,
                          object_points, K, D, args)
        else:
            run_interactive(cv2, sl, camera, runtime, mat, info, aruco, detector, dictionary, params,
                            object_points, K, D, args, headless)
    finally:
        camera.close()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.')
