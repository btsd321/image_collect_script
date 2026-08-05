#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 captured_images 文件夹转换为本工程 bbox3d 标注所需的 3D 文件夹格式。

输入目录结构（ROS CameraInfo 风格）::

    <INPUT>/
    ├── rgb/          # *.png
    ├── depth/        # *.png (uint16=mm / float32=m，由 dtype 推断单位)
    └── camera_info/  # *.json (ROS sensor_msgs/CameraInfo)

输出目录结构（本工程 `open_3d_folder_dialog` 约定）::

    <OUTPUT>/
    ├── rgb/          # 原样复制
    ├── depth/        # 原样复制
    └── camera_info/  # 转换后的内参 json（与本工程 bbox3d 格式一致）

要求同一个样本的 rgb / depth / camera_info 文件名完全一致。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 本工程约定：depth dtype 推断单位
#   uint16  -> 毫米 (mm)
#   float32 -> 米 (m)
# 这里只做复制，不改 dtype，标注端读取时按本约定推断。

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _parse_k(k: List[float]) -> List[List[float]]:
    """将 ROS CameraInfo.k（长度 9 的行优先一维数组）转为 3x3 二维数组。"""
    if len(k) != 9:
        raise ValueError(
            f"CameraInfo.k 长度应为 9，实际 {len(k)}"
        )
    return [list(k[i * 3 : i * 3 + 3]) for i in range(3)]


def convert_camera_info(
    src: Dict[str, Any],
    camera: Optional[str] = None,
) -> Dict[str, Any]:
    """把 ROS CameraInfo 字典转换为本工程 bbox3d 内参 json 格式。

    ROS 字段映射：
      height/width -> image_height/image_width
      k            -> intrinsic_matrix (3x3) + fx/fy/cx/cy
      d            -> distortion.coefficients
      distortion_model -> distortion.model
    """
    height = src.get("height")
    width = src.get("width")
    k = src.get("k", [])
    d = src.get("d", [])
    model = src.get("distortion_model", "plumb_bob")

    if height is None or width is None:
        raise ValueError("CameraInfo 缺少 height/width 字段")

    matrix = _parse_k(k)
    fx = matrix[0][0]
    fy = matrix[1][1]
    cx = matrix[0][2]
    cy = matrix[1][2]

    out: Dict[str, Any] = {
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "intrinsic_matrix": matrix,
            "image_width": int(width),
            "image_height": int(height),
        },
        "distortion": {
            "model": model,
            "coefficients": list(d),
        },
    }
    if camera:
        out["camera"] = camera
    # camera 字段放在最前，符合本工程示例顺序
    if camera:
        out = {"camera": camera, **out}
    return out


def _list_samples(
    rgb_dir: Path,
    depth_dir: Path,
    cam_dir: Path,
) -> List[str]:
    """收集三类文件都存在的样本名（不带扩展名）。

    优先以 rgb 为基准，要求 depth、camera_info 各有同名文件。
    文件名以 rgb 的扩展名为准（rgb 可能是 .png，camera_info 是 .json）。
    """
    rgb_samples = {
        p.stem: p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    }
    depth_samples = {
        p.stem: p for p in depth_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    }
    cam_samples = {
        p.stem: p for p in cam_dir.iterdir() if p.suffix == ".json"
    }

    samples = sorted(
        set(rgb_samples) & set(depth_samples) & set(cam_samples)
    )
    # 顺便报告缺失项
    missing_depth = sorted(set(rgb_samples) - set(depth_samples))
    missing_cam = sorted(set(rgb_samples) - set(cam_samples))
    extra_depth = sorted(set(depth_samples) - set(rgb_samples))
    extra_cam = sorted(set(cam_samples) - set(rgb_samples))
    if missing_depth:
        print(
            f"[WARN] {len(missing_depth)} 个 rgb 样本缺 depth，"
            f"示例：{missing_depth[:3]}"
        )
    if missing_cam:
        print(
            f"[WARN] {len(missing_cam)} 个 rgb 样本缺 camera_info，"
            f"示例：{missing_cam[:3]}"
        )
    if extra_depth:
        print(
            f"[WARN] {len(extra_depth)} 个 depth 无对应 rgb，已忽略，"
            f"示例：{extra_depth[:3]}"
        )
    if extra_cam:
        print(
            f"[WARN] {len(extra_cam)} 个 camera_info 无对应 rgb，已忽略，"
            f"示例：{extra_cam[:3]}"
        )
    return samples


def convert(
    input_dir: Path,
    output_dir: Path,
    camera: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    rgb_dir = input_dir / "rgb"
    depth_dir = input_dir / "depth"
    cam_dir = input_dir / "camera_info"

    for d in (rgb_dir, depth_dir, cam_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"输入目录不存在：{d}")

    out_rgb = output_dir / "rgb"
    out_depth = output_dir / "depth"
    out_cam = output_dir / "camera_info"
    for d in (out_rgb, out_depth, out_cam):
        d.mkdir(parents=True, exist_ok=True)

    samples = _list_samples(rgb_dir, depth_dir, cam_dir)
    if not samples:
        print("[ERROR] 没有找到 rgb/depth/camera_info 三类都存在的样本")
        sys.exit(1)

    print(
        f"[INFO] 共 {len(samples)} 个样本 -> {output_dir} "
        f"(overwrite={overwrite})"
    )

    n_ok = 0
    n_skip = 0
    n_fail = 0
    for stem in samples:
        # 定位真实扩展名
        rgb_src = next(
            (rgb_dir / f"{stem}{e}" for e in IMAGE_EXTS
             if (rgb_dir / f"{stem}{e}").exists()),
            None,
        )
        depth_src = next(
            (depth_dir / f"{stem}{e}" for e in IMAGE_EXTS
             if (depth_dir / f"{stem}{e}").exists()),
            None,
        )
        cam_src = cam_dir / f"{stem}.json"

        rgb_dst = out_rgb / rgb_src.name
        depth_dst = out_depth / depth_src.name
        cam_dst = out_cam / f"{stem}.json"

        # 已存在且不覆盖 -> 跳过
        if (
            not overwrite
            and rgb_dst.exists()
            and depth_dst.exists()
            and cam_dst.exists()
        ):
            n_skip += 1
            continue

        try:
            shutil.copy2(rgb_src, rgb_dst)
            shutil.copy2(depth_src, depth_dst)
            with open(cam_src, "r", encoding="utf-8") as f:
                src_cam = json.load(f)
            dst_cam = convert_camera_info(src_cam, camera=camera)
            with open(cam_dst, "w", encoding="utf-8") as f:
                json.dump(dst_cam, f, indent=2, ensure_ascii=False)
                f.write("\n")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[FAIL] {stem}: {e}")

    print(f"[DONE] 复制/转换成功 {n_ok}，跳过 {n_skip}，失败 {n_fail}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "将 captured_images 文件夹转换为本工程 bbox3d 3D 文件夹格式。"
            " 自动复制 rgb/depth 并将 ROS CameraInfo 转为本工程内参 json。"
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        help="输入 captured_images 文件夹（含 rgb/ depth/ camera_info/）",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="输出 3D 文件夹",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="可选 camera 字段（如 zed / percipio）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出文件",
    )
    args = parser.parse_args()

    if not args.input_dir or not args.output_dir:
        parser.error("请提供 input_dir 和 output_dir")

    convert(
        args.input_dir,
        args.output_dir,
        camera=args.camera,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
