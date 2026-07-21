#!/usr/bin/env python3
"""Data contracts for the RENEE perception pipeline (RGB-D -> point cloud).

Serialization policy:
  - JSON for metadata and the small, fixed-size pose/intrinsic matrices
    (4x4 / 3x3). These round-trip losslessly through nested lists.
  - Large bulk arrays (fused point clouds, meshes) are NEVER embedded:
    contracts reference them by file path (.ply). Per-keyframe RGB/depth
    images are likewise referenced by path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Keyframe", "ScanSession"]


def _arr_to_list(a: np.ndarray | None) -> list | None:
    if a is None:
        return None
    return np.asarray(a, dtype=float).tolist()


def _list_to_arr(x: Any, dtype: type = float) -> np.ndarray | None:
    if x is None:
        return None
    return np.asarray(x, dtype=dtype)


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Keyframe:
    """A single RGB-D capture.

    ``T_world_cam`` is ``None`` until a pose has been estimated (see
    ``perception.odometry``) — raw captures have no pose attached.
    """

    rgb_path: str  # PNG
    depth_path: str  # 16-bit PNG, millimetres
    intrinsics: np.ndarray  # 3x3 pinhole camera matrix
    station_id: int = 0
    timestamp: float = 0.0
    T_world_cam: np.ndarray | None = None  # 4x4 camera pose in world frame

    def to_dict(self) -> dict:
        return {
            "rgb_path": self.rgb_path,
            "depth_path": self.depth_path,
            "intrinsics": _arr_to_list(self.intrinsics),
            "station_id": int(self.station_id),
            "timestamp": float(self.timestamp),
            "T_world_cam": _arr_to_list(self.T_world_cam),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Keyframe":
        return cls(
            rgb_path=d["rgb_path"],
            depth_path=d["depth_path"],
            intrinsics=_list_to_arr(d["intrinsics"]),
            station_id=int(d.get("station_id", 0)),
            timestamp=float(d.get("timestamp", 0.0)),
            T_world_cam=_list_to_arr(d.get("T_world_cam")),
        )


@dataclass
class ScanSession:
    """A set of registered (or not-yet-registered) RGB-D keyframes."""

    session_dir: str
    keyframes: list[Keyframe]
    cloud_path: str | None = None  # fused point cloud (.ply), set after fusion
    mesh_path: str | None = None  # extracted mesh (.ply), set after fusion
    meta: dict = field(default_factory=dict)  # date, camera, capture settings, ...

    def to_dict(self) -> dict:
        return {
            "session_dir": self.session_dir,
            "keyframes": [k.to_dict() for k in self.keyframes],
            "cloud_path": self.cloud_path,
            "mesh_path": self.mesh_path,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScanSession":
        return cls(
            session_dir=d["session_dir"],
            keyframes=[Keyframe.from_dict(k) for k in d["keyframes"]],
            cloud_path=d.get("cloud_path"),
            mesh_path=d.get("mesh_path"),
            meta=d.get("meta", {}),
        )

    def save(self, path: str | Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "ScanSession":
        return cls.from_dict(_read_json(path))
