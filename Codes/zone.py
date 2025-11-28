#!/usr/bin/env python3
"""
shape_detector_task2a.py
----------------------------------------------------
Detect geometric shapes (Triangle, Square, Pentagon)
using LiDAR scan data for eYantra Task 2A.

Publishes:
    /shape_detected (String): "STOP" or "RESUME"
    /detection_status (String): "STATUS,x,y"
----------------------------------------------------
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import math
import time
from sklearn.linear_model import RANSACRegressor

class ShapeDetector(Node):
    def __init__(self):
        super().__init__('shape_detector_task2a')

        # --- Subscribers ---
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # --- Publishers ---
        self.shape_pub = self.create_publisher(String, '/shape_detected', 10)
        self.status_pub = self.create_publisher(String, '/detection_status', 10)

        # --- Robot pose ---
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # --- Detection control ---
        self.last_detection_time = 0.0
        self.detection_cooldown = 5.0  # seconds between detections
        self.min_points = 10
        self.range_min = 0.2
        self.range_max = 2.0

        self.get_logger().info("✅ Shape Detector Node Started")

    # ---------------------------------------------
    def odom_callback(self, msg):
        """Update robot odometry for status publishing"""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    # ---------------------------------------------
    def scan_callback(self, scan):
        """Process LiDAR data for shape detection"""
        now = time.time()
        if now - self.last_detection_time < self.detection_cooldown:
            return  # Skip until cooldown finishes

        ranges = np.array(scan.ranges)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))
        valid = (ranges > self.range_min) & (ranges < self.range_max)
        ranges, angles = ranges[valid], angles[valid]

        if len(ranges) < self.min_points:
            return

        # Convert to Cartesian
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        points = np.vstack((xs, ys)).T

        # --- Step 1: Segment lines ---
        lines = self.extract_lines(points)
        if len(lines) < 3:
            return  # Not enough edges for a polygon

        # --- Step 2: Polygon classification ---
        polygon_type = self.classify_polygon(lines)

        if polygon_type:
            # --- Step 3: Publish detection ---
            self.handle_detection(polygon_type)

    # ---------------------------------------------
    def extract_lines(self, points, dist_thresh=0.05, min_points=8):
        """Simple line extraction using RANSAC"""
        lines = []
        remaining = points.copy()
        while len(remaining) > min_points:
            X = remaining[:, 0].reshape(-1, 1)
            y = remaining[:, 1]
            model = RANSACRegressor(residual_threshold=dist_thresh)
            model.fit(X, y)
            inlier_mask = model.inlier_mask_
            inliers = remaining[inlier_mask]
            if len(inliers) > min_points:
                lines.append(inliers)
            remaining = remaining[~inlier_mask]
        return lines

    # ---------------------------------------------
    def classify_polygon(self, lines):
        """Classify polygon based on number of line segments and angles"""
        num_lines = len(lines)

        # Merge close/collinear lines
        if num_lines in [3, 4, 5]:
            # Estimate average angle difference
            angles = []
            for seg in lines:
                x1, y1 = seg[0]
                x2, y2 = seg[-1]
                angles.append(math.atan2(y2 - y1, x2 - x1))
            angles = np.unwrap(angles)
            total_angle_span = abs(angles.max() - angles.min())

            # Approximate by count of edges
            if num_lines == 3:
                return "FERTILIZER_REQUIRED"  # Triangle
            elif num_lines == 4:
                return "BAD_HEALTH"  # Square
            elif num_lines == 5:
                return "DOCK_STATION"  # Pentagon
        return None

    # ---------------------------------------------
    def handle_detection(self, shape_status):
        """Publish shape detection info"""
        self.last_detection_time = time.time()

        # 1️⃣ Publish STOP
        stop_msg = String()
        stop_msg.data = "STOP"
        self.shape_pub.publish(stop_msg)

        # 2️⃣ Publish Detection Status
        status_msg = String()
        status_msg.data = f"{shape_status},{self.x:.2f},{self.y:.2f}"
        self.status_pub.publish(status_msg)

        self.get_logger().info(f"📍 Detected: {status_msg.data}")
        self.get_logger().info("⏸️ Holding for 2 seconds...")
        time.sleep(2.0)

        # 3️⃣ Publish RESUME
        resume_msg = String()
        resume_msg.data = "RESUME"
        self.shape_pub.publish(resume_msg)

        self.get_logger().info("✅ Resuming navigation\n")

    # ---------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
