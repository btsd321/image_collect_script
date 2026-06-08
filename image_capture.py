#!/usr/bin/env python3
"""
ROS2 image capture node: displays RGB and depth images, saves on key '1'.
"""

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="ROS2 RGB + Depth image viewer and capture tool")
    parser.add_argument("--rgb-topic", default="/zed/zed_node/left/color/rect/image",
                        help="RGB image topic name")
    parser.add_argument("--depth-topic", default="/zed/zed_node/depth/depth_registered",
                        help="Depth image topic name")
    parser.add_argument("--camera-info-topic", default="/zed/zed_node/left/color/rect/camera_info",
                        help="CameraInfo topic name")
    parser.add_argument("--output-dir", default="./captured_images",
                        help="Directory to save captured images")
    parser.add_argument("--depth-scale", type=float, default=1000.0,
                        help="Depth scale factor for visualization (mm to m if 1000)")
    parser.add_argument("--window-width", type=int, default=640,
                        help="Display window width per image")
    parser.add_argument("--window-height", type=int, default=360,
                        help="Display window height per image")
    parser.add_argument("--qos-reliability", choices=["reliable", "best_effort"], default="best_effort",
                        help="QoS reliability for subscribers")
    return parser.parse_args()


class ImageCaptureNode:
    def __init__(self, args):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from sensor_msgs.msg import Image, CameraInfo
        try:
            from cv_bridge import CvBridge
            self._bridge = CvBridge()
        except ImportError:
            self._bridge = None

        self._args = args
        self._rgb_frame = None
        self._depth_frame = None
        self._camera_info = None
        self._lock = threading.Lock()
        self._capture_requested = False
        self._capture_count = 0

        # Resolve to absolute path so logs are unambiguous regardless of cwd
        args.output_dir = os.path.abspath(args.output_dir)
        rgb_dir = os.path.join(args.output_dir, "rgb")
        depth_dir = os.path.join(args.output_dir, "depth")
        camera_info_dir = os.path.join(args.output_dir, "camera_info")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(camera_info_dir, exist_ok=True)
        print(f"[DEBUG] cwd            = {os.getcwd()}")
        print(f"[DEBUG] output_dir abs = {args.output_dir}")
        print(f"[DEBUG] rgb_dir   abs  = {rgb_dir}  exists={os.path.isdir(rgb_dir)}  writable={os.access(rgb_dir, os.W_OK)}")
        print(f"[DEBUG] depth_dir abs  = {depth_dir}  exists={os.path.isdir(depth_dir)}  writable={os.access(depth_dir, os.W_OK)}")
        print(f"[DEBUG] camera_info_dir abs = {camera_info_dir}  exists={os.path.isdir(camera_info_dir)}  writable={os.access(camera_info_dir, os.W_OK)}")

        reliability = (ReliabilityPolicy.BEST_EFFORT
                       if args.qos_reliability == "best_effort"
                       else ReliabilityPolicy.RELIABLE)
        qos = QoSProfile(
            reliability=reliability,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        rclpy.init()
        self._node = Node("image_capture_node")
        self._node.create_subscription(Image, args.rgb_topic, self._rgb_callback, qos)
        self._node.create_subscription(Image, args.depth_topic, self._depth_callback, qos)
        self._node.create_subscription(CameraInfo, args.camera_info_topic, self._camera_info_callback, qos)

        self._node.get_logger().info(f"Subscribing RGB:   {args.rgb_topic}")
        self._node.get_logger().info(f"Subscribing Depth: {args.depth_topic}")
        self._node.get_logger().info(f"Subscribing CameraInfo: {args.camera_info_topic}")
        self._node.get_logger().info(f"Save directory:    {args.output_dir}")
        self._node.get_logger().info("Press '1' in the display window to capture images. Press 'q' or ESC to quit.")

    def _ros_image_to_cv2(self, msg, expect: str = "any"):
        """Convert sensor_msgs/Image to OpenCV mat without cv_bridge dependency.

        expect: 'rgb' → return BGR uint8; 'depth' → return raw passthrough (preserve dtype/channels);
                'any' → passthrough.
        """
        if self._bridge is not None:
            try:
                if expect == "rgb":
                    return self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                # depth / any: never convert color space, preserve original dtype + channels
                return self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            except Exception:
                pass

        # Manual fallback conversion
        dtype_map = {
            "rgb8":   (np.uint8, 3),
            "bgr8":   (np.uint8, 3),
            "rgba8":  (np.uint8, 4),
            "bgra8":  (np.uint8, 4),
            "mono8":  (np.uint8, 1),
            "mono16": (np.uint16, 1),
            "16UC1":  (np.uint16, 1),
            "32FC1":  (np.float32, 1),
        }
        encoding = msg.encoding
        if encoding not in dtype_map:
            raise ValueError(f"Unsupported encoding: {encoding}")
        dtype, channels = dtype_map[encoding]
        img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)
        if encoding in ("rgb8", "rgba8"):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if channels == 3 else cv2.COLOR_RGBA2BGRA)
        return img

    def _rgb_callback(self, msg):
        try:
            img = self._ros_image_to_cv2(msg, expect="rgb")
            with self._lock:
                if self._rgb_frame is None:
                    self._node.get_logger().info(
                        f"First RGB frame: encoding={msg.encoding}, shape={img.shape}, dtype={img.dtype}")
                self._rgb_frame = img.copy()
        except Exception as e:
            self._node.get_logger().warn(f"RGB conversion error: {e}")

    def _depth_callback(self, msg):
        try:
            img = self._ros_image_to_cv2(msg, expect="depth")
            # 深度必须是单通道；多通道说明 encoding 走错（XYZ 点云图、RGB 错配等）
            if img.ndim == 3 and img.shape[2] != 1:
                self._node.get_logger().warn(
                    f"Depth has {img.shape[2]} channels (encoding={msg.encoding}). "
                    f"Expected 1-channel 16UC1(mm) or 32FC1(m). Using channel 0 only — "
                    f"verify the topic outputs raw depth, not point cloud / RGB.")
                img = img[:, :, 0]
            with self._lock:
                if self._depth_frame is None:
                    self._node.get_logger().info(
                        f"First depth frame: encoding={msg.encoding}, shape={img.shape}, dtype={img.dtype}, "
                        f"min={float(np.nanmin(img)):.4f}, max={float(np.nanmax(img)):.4f}")
                self._depth_frame = img.copy()
        except Exception as e:
            self._node.get_logger().warn(f"Depth conversion error: {e}")

    def _camera_info_callback(self, msg):
        """Store the latest CameraInfo message."""
        with self._lock:
            self._camera_info = msg

    def _depth_to_colormap(self, depth_img):
        """Convert raw depth image to a visible colormap."""
        if depth_img.dtype == np.float32:
            # Replace NaN/inf with 0
            depth_clean = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
            # Normalize to 0-255 for display (clip at 10m)
            depth_norm = np.clip(depth_clean / 10.0, 0, 1)
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
        elif depth_img.dtype == np.uint16:
            depth_uint8 = (depth_img / self._args.depth_scale * 25.5).clip(0, 255).astype(np.uint8)
        else:
            depth_uint8 = depth_img.astype(np.uint8)

        if len(depth_uint8.shape) == 3:
            depth_uint8 = depth_uint8[:, :, 0]
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

    def _save_images(self, rgb, depth, camera_info):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self._capture_count += 1
        idx = self._capture_count
        filename_base = f"{idx:04d}_{ts}"
        filename_img = f"{filename_base}.png"
        filename_json = f"{filename_base}.json"

        rgb_path   = os.path.join(self._args.output_dir, "rgb",   filename_img)
        depth_path = os.path.join(self._args.output_dir, "depth", filename_img)
        camera_info_path = os.path.join(self._args.output_dir, "camera_info", filename_json)

        log = self._node.get_logger()
        log.info(f"[Capture #{idx}] rgb   shape={rgb.shape} dtype={rgb.dtype}  -> {rgb_path}")
        log.info(f"[Capture #{idx}] depth shape={depth.shape} dtype={depth.dtype}")

        # ── RGB ───────────────────────────────────────────────────────────────
        try:
            ok_rgb = cv2.imwrite(rgb_path, rgb)
        except cv2.error as e:
            ok_rgb = False
            log.error(f"[Capture #{idx}] cv2.imwrite RGB threw: {e}")
        if ok_rgb and os.path.isfile(rgb_path):
            sz = os.path.getsize(rgb_path)
            log.info(f"[Capture #{idx}] RGB   ok  ({sz} bytes)")
        else:
            log.error(f"[Capture #{idx}] RGB   FAILED  (imwrite={ok_rgb}, exists={os.path.isfile(rgb_path)}, "
                      f"dir_writable={os.access(os.path.dirname(rgb_path), os.W_OK)})")

        # ── Depth: always save as single-channel uint16 PNG, unit = mm ──────
        # 防御：上游若漏掉单通道约束，这里再 squeeze 一次（理论上已被 _depth_callback 拦截）
        if depth.ndim == 3:
            if depth.shape[2] == 1:
                depth = depth[:, :, 0]
            else:
                log.warn(f"[Capture #{idx}] depth still has {depth.shape[2]} channels after callback "
                         f"sanitization; saving channel 0 only.")
                depth = depth[:, :, 0]

        if depth.dtype == np.float32:
            # 32FC1: values in metres → convert to mm
            depth_mm = (np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0)
        elif depth.dtype == np.uint16:
            # 16UC1: values already in mm (ZED SDK convention)
            depth_mm = depth.astype(np.float32)
        else:
            log.warn(f"[Capture #{idx}] unexpected depth dtype={depth.dtype}; "
                     f"casting to float32 without unit conversion (verify result).")
            depth_mm = depth.astype(np.float32)

        depth_uint16 = depth_mm.clip(0, 65535).astype(np.uint16)
        log.info(f"[Capture #{idx}] depth_uint16 shape={depth_uint16.shape} dtype={depth_uint16.dtype} "
                 f"min={int(depth_uint16.min())} max={int(depth_uint16.max())}")

        try:
            ok_depth = cv2.imwrite(depth_path, depth_uint16)
        except cv2.error as e:
            ok_depth = False
            log.error(f"[Capture #{idx}] cv2.imwrite Depth threw: {e}")
        if ok_depth and os.path.isfile(depth_path):
            sz = os.path.getsize(depth_path)
            log.info(f"[Capture #{idx}] Depth ok  ({sz} bytes)")
        else:
            log.error(f"[Capture #{idx}] Depth FAILED  (imwrite={ok_depth}, exists={os.path.isfile(depth_path)}, "
                      f"dir_writable={os.access(os.path.dirname(depth_path), os.W_OK)})")

        # ── CameraInfo: save as JSON ──────────────────────────────────────────
        if camera_info is not None:
            camera_info_dict = {
                "header": {
                    "stamp": {
                        "sec": camera_info.header.stamp.sec,
                        "nanosec": camera_info.header.stamp.nanosec,
                    },
                    "frame_id": camera_info.header.frame_id,
                },
                "height": camera_info.height,
                "width": camera_info.width,
                "distortion_model": camera_info.distortion_model,
                "d": list(camera_info.d),
                "k": list(camera_info.k),
                "r": list(camera_info.r),
                "p": list(camera_info.p),
                "binning_x": camera_info.binning_x,
                "binning_y": camera_info.binning_y,
                "roi": {
                    "x_offset": camera_info.roi.x_offset,
                    "y_offset": camera_info.roi.y_offset,
                    "height": camera_info.roi.height,
                    "width": camera_info.roi.width,
                    "do_rectify": camera_info.roi.do_rectify,
                }
            }
            try:
                with open(camera_info_path, 'w') as f:
                    json.dump(camera_info_dict, f, indent=2)
                if os.path.isfile(camera_info_path):
                    sz = os.path.getsize(camera_info_path)
                    log.info(f"[Capture #{idx}] CameraInfo ok  ({sz} bytes)")
                else:
                    log.error(f"[Capture #{idx}] CameraInfo FAILED (file does not exist after write)")
            except Exception as e:
                log.error(f"[Capture #{idx}] CameraInfo save error: {e}")
        else:
            log.warn(f"[Capture #{idx}] CameraInfo not available, skipping JSON save")

    def _spin_thread(self):
        import rclpy
        rclpy.spin(self._node)

    def run(self):
        spin_thread = threading.Thread(target=self._spin_thread, daemon=True)
        spin_thread.start()

        w, h = self._args.window_width, self._args.window_height
        cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RGB", w, h)

        print("Display window ready. Press '1' to capture RGB+Depth, 'q'/ESC to quit.")

        try:
            while True:
                with self._lock:
                    rgb = self._rgb_frame.copy() if self._rgb_frame is not None else None
                    depth = self._depth_frame.copy() if self._depth_frame is not None else None
                    camera_info = self._camera_info

                if rgb is not None:
                    rgb_display = cv2.resize(rgb, (w, h))
                    status = f"Depth: {'OK' if depth is not None else 'waiting...'}  CameraInfo: {'OK' if camera_info is not None else 'waiting...'}"
                    cv2.putText(rgb_display, "Press '1' to capture  " + status, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("RGB", rgb_display)
                else:
                    placeholder = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Waiting for RGB topic...", (10, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
                    cv2.imshow("RGB", placeholder)

                key = cv2.waitKey(30) & 0xFF

                # Window close button: cv2.getWindowProperty returns -1 when destroyed
                if cv2.getWindowProperty("RGB", cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed, quitting...")
                    break

                if key == ord('1'):
                    if rgb is not None and depth is not None and camera_info is not None:
                        self._save_images(rgb, depth, camera_info)
                    else:
                        missing = []
                        if rgb is None:
                            missing.append("RGB")
                        if depth is None:
                            missing.append("Depth")
                        if camera_info is None:
                            missing.append("CameraInfo")
                        print(f"Cannot capture: waiting for {', '.join(missing)} topic(s)")

                elif key in (ord('q'), 27):  # 'q' or ESC
                    print("Quitting...")
                    break

        except KeyboardInterrupt:
            print("\nInterrupted, quitting...")
        finally:
            cv2.destroyAllWindows()
            self._node.destroy_node()
            import rclpy
            rclpy.shutdown()


def main():
    args = parse_args()
    node = ImageCaptureNode(args)
    node.run()


if __name__ == "__main__":
    main()
