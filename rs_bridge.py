#!/usr/bin/env python3
"""RealSense frame bridge — captures via librealsense (ROS2 node) and writes to shared volume.

This runs on the HOST (not in Docker). It:
1. Launches the RealSense camera via ROS2 realsense2_camera package
2. Subscribes to the color image topic
3. Writes the latest 640x480 BGR frame to /tmp/inference_diag/rs_top.npy
4. The Docker inference client reads from that file for the top camera

Usage:
    source /opt/ros/jazzy/setup.bash
    python3 rs_bridge.py
"""
import os, sys, time, signal
import numpy as np

os.environ.setdefault("ROS_DOMAIN_ID", "0")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

FRAME_PATH = "/tmp/inference_diag/rs_top.npy"
FRAME_TMP  = "/tmp/inference_diag/rs_top.tmp.npy"

class RsBridge(Node):
    def __init__(self):
        super().__init__("rs_bridge")
        self.bridge = CvBridge()
        self.count = 0
        self.sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.on_image,
            10
        )
        self.get_logger().info("Waiting for RealSense color frames on /camera/camera/color/image_raw ...")

    def on_image(self, msg):
        # RGB matches the training data (so101_ros2/camera_streams.py uses rgb8)
        # and matches how the docker client consumes V4L2 frames (BGR→RGB).
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        # Resize to 640x480 if needed
        h, w = frame.shape[:2]
        if (w, h) != (640, 480):
            frame = cv2.resize(frame, (640, 480))
        # Atomic write: write to tmp, rename
        np.save(FRAME_TMP, frame)
        os.replace(FRAME_TMP, FRAME_PATH)
        self.count += 1
        if self.count % 30 == 1:
            self.get_logger().info(f"Frame {self.count}: {frame.shape} RGB mean={frame.mean():.0f}")

def main():
    os.makedirs("/tmp/inference_diag", exist_ok=True)
    rclpy.init()
    node = RsBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
