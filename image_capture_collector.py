#!/usr/bin/env python3
"""
图像采集数据收集脚本

从 image_capture.py 收集的数据中提取 RGB、Depth 和相机内参，重组织为训练数据集。

用法:
    python image_capture_collector.py \
        --input ./captured_images \
        --output ./training_dataset \
        --deduplicate \
        --hamming-threshold 0.05
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def compute_image_hash(image: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """
    计算图像的感知哈希（pHash）

    Args:
        image: 输入图像（BGR 或灰度）
        hash_size: 哈希尺寸（默认 16x16）

    Returns:
        二值哈希数组（flatten，256 位）
    """
    # 转换为灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Resize 到 hash_size x hash_size
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)

    # DCT 变换（取左上角低频部分）
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:8, :8]

    # 计算均值，二值化
    median = np.median(dct_low)
    hash_bits = (dct_low > median).astype(np.uint8)

    return hash_bits.flatten()


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> float:
    """
    计算两个哈希的汉明距离（归一化到 [0, 1]）

    Args:
        hash1, hash2: 二值哈希数组

    Returns:
        归一化汉明距离（0=完全相同，1=完全不同）
    """
    return np.count_nonzero(hash1 != hash2) / len(hash1)


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    解析 image_capture.py 生成的文件名格式: {idx:04d}_{timestamp}.ext

    Args:
        filename: 文件名（例如 0001_20240803_123456_789.png）

    Returns:
        包含 idx 和 timestamp 的字典，解析失败返回 None
    """
    # 匹配格式: 0001_20240803_123456_789.png
    pattern = r"^(\d{4})_(.+)\.(png|json)$"
    match = re.match(pattern, filename)

    if not match:
        return None

    idx, timestamp, ext = match.groups()
    return {
        "idx": idx,
        "timestamp": timestamp,
        "ext": ext,
        "base": f"{idx}_{timestamp}"
    }


def find_rgb_files(input_rgb_dir: Path) -> List[Dict]:
    """
    查找所有 RGB 图像文件并解析文件名

    Returns:
        包含文件路径和解析信息的字典列表
    """
    if not input_rgb_dir.exists():
        print(f"[ERROR] RGB 目录不存在: {input_rgb_dir}", file=sys.stderr)
        return []

    rgb_files = []
    for rgb_path in sorted(input_rgb_dir.glob("*.png")):
        parsed = parse_filename(rgb_path.name)
        if parsed is None:
            print(f"[WARN] 跳过无法解析的文件名: {rgb_path.name}", file=sys.stderr)
            continue

        rgb_files.append({
            "rgb_path": rgb_path,
            "timestamp": parsed["timestamp"],
            "base": parsed["base"]
        })

    return rgb_files


def convert_camera_info(ros_camera_info: Dict) -> Dict:
    """
    转换 ROS CameraInfo 格式到 train_collector 格式

    Args:
        ros_camera_info: ROS sensor_msgs/CameraInfo 的 JSON 表示

    Returns:
        train_collector 格式的相机信息字典
    """
    # 从 K 矩阵提取内参 [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    k = ros_camera_info.get("k", [])
    if len(k) < 9:
        print(f"[ERROR] 无效的 K 矩阵: {k}", file=sys.stderr)
        return None

    fx, _, cx = k[0], k[1], k[2]
    _, fy, cy = k[3], k[4], k[5]

    intrinsic_matrix = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]

    # 畸变系数
    d = ros_camera_info.get("d", [0.0, 0.0, 0.0, 0.0, 0.0])

    camera_info = {
        "camera": "captured",
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "intrinsic_matrix": intrinsic_matrix,
            "image_width": ros_camera_info.get("width"),
            "image_height": ros_camera_info.get("height")
        },
        "distortion": {
            "model": ros_camera_info.get("distortion_model", "plumb_bob"),
            "coefficients": d
        }
    }

    return camera_info


class SampleRecord:
    """单个样本记录（用于去重）"""
    def __init__(
        self,
        timestamp: str,
        rgb_path: Path,
        depth_path: Optional[Path] = None,
        camera_info_path: Optional[Path] = None
    ):
        self.timestamp = timestamp
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.camera_info_path = camera_info_path

    def compute_hash(self) -> Optional[np.ndarray]:
        """按需计算哈希（不缓存，避免内存累积）"""
        if not self.rgb_path.exists():
            return None
        try:
            img = cv2.imread(str(self.rgb_path))
            if img is None:
                return None
            return compute_image_hash(img)
        except Exception as e:
            print(f"[ERROR] 计算哈希失败 {self.rgb_path}: {e}", file=sys.stderr)
            return None


def deduplicate_samples(
    samples: List[SampleRecord],
    hamming_threshold: float
) -> List[SampleRecord]:
    """
    对样本进行去重（时间戳排序，流式相邻汉明距离去重）

    内存优化：按需计算哈希，不缓存图像，仅保留当前和前一个哈希

    Args:
        samples: 样本记录列表
        hamming_threshold: 汉明距离阈值（低于此值认为重复）

    Returns:
        去重后的样本列表
    """
    if not samples:
        return []

    print(f"\n[INFO] 去重样本 ({len(samples)} 个)...")

    # 按时间戳排序
    sorted_samples = sorted(samples, key=lambda s: s.timestamp)

    # 流式去重：仅保留前一个哈希
    keep = []
    prev_hash = None

    for current in sorted_samples:
        # 按需计算当前样本哈希
        current_hash = current.compute_hash()

        if current_hash is None:
            # 无哈希，跳过去重直接保留
            keep.append(current)
            prev_hash = None  # 重置前一个哈希
            continue

        # 检查是否与前一个样本重复
        if prev_hash is not None:
            dist = hamming_distance(prev_hash, current_hash)
            if dist < hamming_threshold:
                # 重复：替换为更新的样本（当前样本时间戳更大）
                print(f"[DEDUP] 丢弃 {keep[-1].timestamp} (汉明距离 {dist:.3f} < {hamming_threshold})")
                keep[-1] = current
                prev_hash = current_hash  # 更新前一个哈希
                continue

        # 不重复，保留
        keep.append(current)
        prev_hash = current_hash

    print(f"[INFO] 去重完成: {len(sorted_samples)} -> {len(keep)} ({len(sorted_samples) - len(keep)} 个重复)")

    return keep


def process_rgb(sample: SampleRecord, output_rgb_dir: Path) -> bool:
    """处理 RGB 图像：复制并重命名为 {timestamp}.png"""
    output_path = output_rgb_dir / f"{sample.timestamp}.png"

    try:
        shutil.copy2(sample.rgb_path, output_path)
        return True
    except Exception as e:
        print(f"[ERROR] 复制 RGB 失败 {sample.rgb_path} -> {output_path}: {e}", file=sys.stderr)
        return False


def process_depth(sample: SampleRecord, output_depth_dir: Path) -> bool:
    """处理深度图：uint16(mm) → float32(m)，保存为 {timestamp}.npy"""
    if sample.depth_path is None or not sample.depth_path.exists():
        print(f"[WARN] 深度图不存在: {sample.timestamp}", file=sys.stderr)
        return False

    try:
        # 读取深度图（应为 uint16 单通道，单位毫米）
        depth_img = cv2.imread(str(sample.depth_path), cv2.IMREAD_UNCHANGED)

        if depth_img is None:
            print(f"[ERROR] 无法读取深度图: {sample.depth_path}", file=sys.stderr)
            return False

        # 校验格式：期望 uint16 单通道
        if depth_img.dtype != np.uint16:
            print(f"[WARN] 深度图 dtype 非 uint16（实际 {depth_img.dtype}）: {sample.depth_path}",
                  file=sys.stderr)
        if depth_img.ndim != 2:
            print(f"[ERROR] 深度图非单通道（shape={depth_img.shape}），跳过: {sample.depth_path}",
                  file=sys.stderr)
            return False

        # 单位换算：毫米 → 米
        depth_float = depth_img.astype(np.float32) / 1000.0

        # 保存为 .npy
        output_path = output_depth_dir / f"{sample.timestamp}.npy"
        np.save(output_path, depth_float)
        return True
    except Exception as e:
        print(f"[ERROR] 处理 Depth 失败 {sample.depth_path}: {e}", file=sys.stderr)
        return False


def process_camera_info(sample: SampleRecord, output_camera_info_dir: Path) -> bool:
    """提取相机内参和畸变，转换格式并保存为 {timestamp}.json"""
    if sample.camera_info_path is None or not sample.camera_info_path.exists():
        print(f"[WARN] 相机信息不存在: {sample.timestamp}", file=sys.stderr)
        return False

    try:
        # 读取 ROS CameraInfo JSON
        with open(sample.camera_info_path, "r", encoding="utf-8") as f:
            ros_camera_info = json.load(f)

        # 转换格式
        camera_info = convert_camera_info(ros_camera_info)

        if camera_info is None:
            return False

        # 保存
        output_path = output_camera_info_dir / f"{sample.timestamp}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(camera_info, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        print(f"[ERROR] 处理相机信息失败 {sample.camera_info_path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="从 image_capture.py 采集数据中收集训练数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅收集 RGB（默认）
  python image_capture_collector.py \\
      --input ./captured_images \\
      --output ./training_data

  # 收集完整数据（RGB + Depth + CameraInfo）
  python image_capture_collector.py \\
      --input ./captured_images \\
      --output ./training_data \\
      --full

  # 启用去重
  python image_capture_collector.py \\
      --input ./captured_images \\
      --output ./training_data \\
      --deduplicate \\
      --hamming-threshold 0.05
        """
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入文件夹路径（captured_images）"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出文件夹路径"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="收集完整数据（RGB + Depth + CameraInfo）；默认仅收集 RGB"
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="启用去重逻辑（基于 RGB 图像汉明距离）"
    )
    parser.add_argument(
        "--hamming-threshold",
        type=float,
        default=0.05,
        help="汉明距离阈值（默认: 0.05，越小越严格）"
    )

    args = parser.parse_args()

    # 创建输出目录结构
    output_rgb_dir = args.output / "rgb"
    output_rgb_dir.mkdir(parents=True, exist_ok=True)

    if args.full:
        print("[INFO] 完整收集模式（RGB + Depth + CameraInfo）")
        output_depth_dir = args.output / "depth"
        output_camera_info_dir = args.output / "camera_info"
        output_depth_dir.mkdir(parents=True, exist_ok=True)
        output_camera_info_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("[INFO] 仅收集 RGB 模式（默认；加 --full 收集完整数据）")
        output_depth_dir = None
        output_camera_info_dir = None

    # 查找输入目录
    input_rgb_dir = args.input / "rgb"
    input_depth_dir = args.input / "depth"
    input_camera_info_dir = args.input / "camera_info"

    # 查找所有 RGB 文件
    print(f"[INFO] 扫描 {input_rgb_dir} ...")
    rgb_files = find_rgb_files(input_rgb_dir)
    print(f"[INFO] 找到 {len(rgb_files)} 个 RGB 图像")

    if len(rgb_files) == 0:
        print("[ERROR] 未找到任何 RGB 图像，退出", file=sys.stderr)
        sys.exit(1)

    # 第一阶段：收集所有样本记录
    samples = []

    for rgb_info in rgb_files:
        timestamp = rgb_info["timestamp"]
        base = rgb_info["base"]

        # 查找对应的 depth 和 camera_info 文件
        depth_path = None
        camera_info_path = None

        if args.full:
            depth_path = input_depth_dir / f"{base}.png"
            camera_info_path = input_camera_info_dir / f"{base}.json"

            # 检查文件是否存在
            if not depth_path.exists():
                print(f"[WARN] 缺少深度图: {base}.png", file=sys.stderr)
                depth_path = None

            if not camera_info_path.exists():
                print(f"[WARN] 缺少相机信息: {base}.json", file=sys.stderr)
                camera_info_path = None

        # 创建样本记录
        sample = SampleRecord(
            timestamp=timestamp,
            rgb_path=rgb_info["rgb_path"],
            depth_path=depth_path,
            camera_info_path=camera_info_path
        )
        samples.append(sample)

    print(f"[INFO] 收集到 {len(samples)} 个样本")

    # 第二阶段：去重（如果启用）
    if args.deduplicate:
        print(f"\n[INFO] 启用去重逻辑 (汉明距离阈值: {args.hamming_threshold})")
        samples = deduplicate_samples(samples, args.hamming_threshold)
        print(f"[INFO] 去重后剩余 {len(samples)} 个样本")
    else:
        print(f"[INFO] 跳过去重")

    # 第三阶段：处理并保存样本
    print(f"\n[INFO] 开始处理样本...")
    success_count = 0
    fail_count = 0

    for i, sample in enumerate(samples, start=1):
        if (i - 1) % 50 == 0:
            print(f"[{i}/{len(samples)}] 处理样本...")

        # 检查是否已存在（避免重复处理）
        if (output_rgb_dir / f"{sample.timestamp}.png").exists():
            continue

        # 处理 RGB
        rgb_ok = process_rgb(sample, output_rgb_dir)

        # 处理 Depth 和相机信息（仅在 --full 模式）
        if args.full:
            depth_ok = process_depth(sample, output_depth_dir)
            camera_ok = process_camera_info(sample, output_camera_info_dir)
        else:
            depth_ok = True  # 跳过，视为成功
            camera_ok = True

        if rgb_ok and depth_ok and camera_ok:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"[SUMMARY] 成功: {success_count}, 失败: {fail_count}, 总计: {len(samples)}")
    print(f"[OUTPUT] {args.output}")
    print(f"  - rgb/        : {len(list(output_rgb_dir.glob('*.png')))} 张")
    if args.full:
        print(f"  - depth/      : {len(list(output_depth_dir.glob('*.npy')))} 张")
        print(f"  - camera_info/: {len(list(output_camera_info_dir.glob('*.json')))} 个")


if __name__ == "__main__":
    main()
