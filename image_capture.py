#!/usr/bin/env python3
"""
ROS2 image capture node: displays RGB and depth images, saves on key '1'.
"""

import argparse
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
        from sensor_msgs.msg import Image
        try:
            from cv_bridge import CvBridge
            self._bridge = CvBridge()
        except ImportError:
            self._bridge = None

        self._args = args
        self._rgb_frame = None
        self._depth_frame = None
        self._lock = threading.Lock()
        self._capture_requested = False
        self._capture_count = 0

        os.makedirs(os.path.join(args.output_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "depth"), exist_ok=True)

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

        self._node.get_logger().info(f"Subscribing RGB:   {args.rgb_topic}")
        self._node.get_logger().info(f"Subscribing Depth: {args.depth_topic}")
        self._node.get_logger().info(f"Save directory:    {args.output_dir}")
        self._node.get_logger().info("Press '1' in the display window to capture images. Press 'q' or ESC to quit.")

    def _ros_image_to_cv2(self, msg):
        """Convert sensor_msgs/Image to OpenCV mat without cv_bridge dependency."""
        if self._bridge is not None:
            try:
                encoding = msg.encoding
                if encoding in ("32FC1", "16UC1"):
                    return self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                return self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
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
            img = self._ros_image_to_cv2(msg)
            with self._lock:
                self._rgb_frame = img.copy()
        except Exception as e:
            self._node.get_logger().warn(f"RGB conversion error: {e}")

    def _depth_callback(self, msg):
        try:
            img = self._ros_image_to_cv2(msg)
            with self._lock:
                self._depth_frame = img.copy()
        except Exception as e:
            self._node.get_logger().warn(f"Depth conversion error: {e}")

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

    def _save_images(self, rgb, depth):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self._capture_count += 1
        idx = self._capture_count
        filename = f"{idx:04d}_{ts}.png"

        rgb_path   = os.path.join(self._args.output_dir, "rgb",   filename)
        depth_path = os.path.join(self._args.output_dir, "depth", filename)

        cv2.imwrite(rgb_path, rgb)

        # Always save depth as uint16 PNG with unit = mm
        if depth.dtype == np.float32:
            # 32FC1: values in metres → convert to mm
            depth_mm = (np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0)
        elif depth.dtype == np.uint16:
            # 16UC1: values already in mm (ZED SDK convention)
            depth_mm = depth.astype(np.float32)
        else:
            depth_mm = depth.astype(np.float32)

        depth_uint16 = depth_mm.clip(0, 65535).astype(np.uint16)
        cv2.imwrite(depth_path, depth_uint16)

        self._node.get_logger().info(
            f"[Capture #{idx}] Saved:\n  RGB:   {rgb_path}\n  Depth: {depth_path}"
        )

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

                if rgb is not None:
                    rgb_display = cv2.resize(rgb, (w, h))
                    status = f"Depth: {'OK' if depth is not None else 'waiting...'}"
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
                    if rgb is not None and depth is not None:
                        self._save_images(rgb, depth)
                    else:
                        missing = []
                        if rgb is None:
                            missing.append("RGB")
                        if depth is None:
                            missing.append("Depth")
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
