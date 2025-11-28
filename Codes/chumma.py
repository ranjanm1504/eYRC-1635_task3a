#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
import cv2.aruco as aruco
import json

# ============================================
# CONFIGURATION - CHANGE THIS FOR HARDWARE
# ============================================
TEAM_ID = 1635  # Your team ID
USE_HARDWARE = False  # Set to True when testing on real hardware


# ============================================

class EnhancedDetector(Node):
    def __init__(self):
        super().__init__('enhanced_detector')

        # Topic configuration based on environment
        if USE_HARDWARE:
            color_topic = '/camera/camera/color/image_raw'
            depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
            camera_info_topic = '/camera/camera/color/camera_info'
            self.get_logger().info("🔧 HARDWARE MODE - Using real robot topics")
        else:
            color_topic = '/camera/image_raw'
            depth_topic = '/camera/depth/image_raw'
            camera_info_topic = '/camera/camera_info'
            self.get_logger().info("💻 SIMULATION MODE - Using simulation topics")

        # Subscriptions
        self.color_sub = self.create_subscription(
            Image, color_topic, self.color_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10
        )

        # Publishers for JSON data
        self.fruit_positions_pub = self.create_publisher(String, '/_bad_fruit_positions', 10)
        self.fertilizer_info_pub = self.create_publisher(String, '/_fruit_info', 10)

        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None
        self.camera_info_received = False

        # TF broadcaster and buffer for lookups
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera intrinsics (will be updated from camera_info)
        self.fx = 915.3003540039062
        self.fy = 914.0320434570312
        self.cx = 642.724365234375
        self.cy = 361.9780578613281

        # ArUco setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.aruco_params = aruco.DetectorParameters()

        # Tracking detected objects - store their base_link positions
        self.detected_fruits = {}
        self.fertilizer_position = None

        # Store last known positions to republish
        self.last_fruit_positions = {}
        self.last_fertilizer_position = None

        # Timer to republish TF at 10Hz
        self.tf_timer = self.create_timer(0.1, self.publish_stored_tfs)

        # Timer to publish JSON data at 1Hz
        self.json_timer = self.create_timer(1.0, self.publish_json_data)

        # Frame counter for periodic status
        self.frame_count = 0

        self.get_logger().info("╔════════════════════════════════════════╗")
        self.get_logger().info("║  Enhanced Detector - Task 3A          ║")
        self.get_logger().info(f"║  Team ID: {TEAM_ID}".ljust(40) + "║")
        self.get_logger().info(f"║  Mode: {'HARDWARE' if USE_HARDWARE else 'SIMULATION'}".ljust(40) + "║")
        self.get_logger().info("╚════════════════════════════════════════╝")
        self.get_logger().info(f"📡 Subscribed to: {color_topic}")
        self.get_logger().info(f"📡 Subscribed to: {depth_topic}")
        self.get_logger().info(f"📡 Subscribed to: {camera_info_topic}")
        self.get_logger().info(f"📤 Publishing to: /_bad_fruit_positions")
        self.get_logger().info(f"📤 Publishing to: /_fruit_info")

    def camera_info_callback(self, msg):
        """Update camera intrinsics from camera_info topic"""
        if not self.camera_info_received:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_info_received = True
            self.get_logger().info(
                f"📷 Camera intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}"
            )

    def publish_json_data(self):
        """Publish detected objects as JSON strings"""

        # Publish bad fruit positions
        if self.last_fruit_positions:
            fruits_data = []
            for fruit_key, (x, y, z, fruit_id) in self.last_fruit_positions.items():
                fruits_data.append({
                    "id": str(fruit_id),  # Convert to string, not bool!
                    "frame_id": f"{TEAM_ID}_bad_fruit_{fruit_id}",
                    "position": {
                        "x": float(x),
                        "y": float(y),
                        "z": float(z)
                    }
                })

            msg = String()
            msg.data = json.dumps(fruits_data)  # This creates a JSON string
            self.fruit_positions_pub.publish(msg)

        # Publish fertilizer position
        if self.last_fertilizer_position is not None:
            x, y, z = self.last_fertilizer_position
            fertilizer_data = {
                "detected": "true",  # String, not boolean!
                "frame_id": f"{TEAM_ID}_fertilizer_1",
                "position": {
                    "x": float(x),
                    "y": float(y),
                    "z": float(z)
                }
            }

            msg = String()
            msg.data = json.dumps(fertilizer_data)
            self.fertilizer_info_pub.publish(msg)

    def publish_stored_tfs(self):
        """Continuously republish stored TF frames at 10Hz"""
        # Republish fertilizer with correct name
        if self.last_fertilizer_position is not None:
            x, y, z = self.last_fertilizer_position
            self.publish_tf(f"{TEAM_ID}_fertilizer_1", x, y, z)

        # Republish fruit positions
        for fruit_key, (x, y, z, fruit_id) in self.last_fruit_positions.items():
            frame_id = f"{TEAM_ID}_bad_fruit_{fruit_id}"
            self.publish_tf(frame_id, x, y, z)

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def get_3d_position(self, cx_full, cy_full):
        """Convert 2D pixel coordinates to 3D world coordinates in camera frame"""
        if self.depth_image is None:
            return None

        if cy_full >= self.depth_image.shape[0] or cx_full >= self.depth_image.shape[1]:
            return None

        # Sample area around point for more robust depth
        window_size = 3
        y_min = max(0, cy_full - window_size)
        y_max = min(self.depth_image.shape[0], cy_full + window_size)
        x_min = max(0, cx_full - window_size)
        x_max = min(self.depth_image.shape[1], cx_full + window_size)

        depth_region = self.depth_image[y_min:y_max, x_min:x_max]
        valid_depths = depth_region[depth_region > 0]

        if len(valid_depths) == 0:
            return None

        depth_val = np.median(valid_depths)
        depth_m = float(depth_val) / 1000.0 if depth_val > 10 else float(depth_val)

        if depth_m <= 0.1 or depth_m > 2.5:
            return None

        # Camera frame coordinates (optical frame convention: X right, Y down, Z forward)
        X_cam = (cx_full - self.cx) * depth_m / self.fx
        Y_cam = (cy_full - self.cy) * depth_m / self.fy
        Z_cam = depth_m

        return (X_cam, Y_cam, Z_cam)

    def transform_to_base_link(self, x_cam, y_cam, z_cam):
        """Transform coordinates from camera frame to base_link frame"""
        try:
            # Try to get TF transform from camera to base_link
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'camera_link_optical',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            cam_point = np.array([x_cam, y_cam, z_cam, 1.0])

            trans = transform.transform.translation
            rot = transform.transform.rotation

            from scipy.spatial.transform import Rotation as R
            rotation_matrix = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()

            T = np.eye(4)
            T[:3, :3] = rotation_matrix
            T[:3, 3] = [trans.x, trans.y, trans.z]

            base_point = T @ cam_point
            return base_point[:3]

        except Exception as e:
            # Fallback: approximate transformation (camera mounted above base)
            # Typical camera mount for UR5 warehouse setup:
            # - Camera is mounted high, pointing down at the scene
            # - Camera optical frame: X right, Y down, Z forward
            # - Base link frame: X forward, Y left, Z up

            x_base = z_cam  # Forward in camera = forward in base
            y_base = -x_cam  # Right in camera = left in base (negative Y)
            z_base = 0.9 - y_cam  # Camera height minus downward offset

            return np.array([x_base, y_base, z_base])

    def publish_tf(self, frame_id, x, y, z):
        """Publish TF transform"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = frame_id
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def detect_fertilizer_aruco(self, image_disp):
        """Detect fertilizer can using ArUco marker"""
        gray = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                # Draw detected marker
                aruco.drawDetectedMarkers(image_disp, corners)

                # Get marker center
                corner = corners[i][0]
                cx = int(np.mean(corner[:, 0]))
                cy = int(np.mean(corner[:, 1]))

                # Get 3D position in camera frame
                cam_pos = self.get_3d_position(cx, cy)
                if cam_pos is None:
                    continue

                X_cam, Y_cam, Z_cam = cam_pos

                # Transform to base_link frame
                base_pos = self.transform_to_base_link(X_cam, Y_cam, Z_cam)
                X, Y, Z = base_pos

                # Store position for continuous publishing
                self.fertilizer_position = base_pos

                # Check if this is a new detection
                if self.last_fertilizer_position is None:
                    self.get_logger().info(
                        f"✅ FERTILIZER DETECTED: [{X:.3f}, {Y:.3f}, {Z:.3f}] (Marker ID: {marker_id})"
                    )

                self.last_fertilizer_position = (X, Y, Z)

                # Publish TF - correct name format
                frame_id = f"{TEAM_ID}_fertilizer_1"
                self.publish_tf(frame_id, X, Y, Z)

                # Draw annotation
                cv2.circle(image_disp, (cx, cy), 10, (255, 0, 0), -1)
                cv2.putText(image_disp, f"Fertilizer (ID:{marker_id})", (cx - 60, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return image_disp

    def detect_bad_fruits(self, image_disp):
        """Detect bad fruits (white/gray colored) in the tray"""
        h, w, _ = image_disp.shape

        # Left tray ROI
        roi_x1 = int(w * 0.0)
        roi_x2 = int(w * 0.35)
        roi_y1 = int(h * 0.25)
        roi_y2 = int(h * 0.58)
        roi = self.color_image[roi_y1:roi_y2, roi_x1:roi_x2].copy()

        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # White/gray fruit detection
        lower_white = np.array([0, 0, 80])
        upper_white = np.array([180, 60, 240])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Morphological operations
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Current frame detections
        current_fruits = {}

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 600 or area > 10000:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Remap to full image coordinates
            cx_full = cx + roi_x1
            cy_full = cy + roi_y1

            # Get 3D position in camera frame
            cam_pos = self.get_3d_position(cx_full, cy_full)
            if cam_pos is None:
                continue

            X_cam, Y_cam, Z_cam = cam_pos

            # Transform to base_link frame
            base_pos = self.transform_to_base_link(X_cam, Y_cam, Z_cam)
            X, Y, Z = base_pos

            # Create unique identifier for this fruit (rounded to 10cm precision)
            fruit_key = (round(X, 1), round(Y, 1), round(Z, 1))

            # Check if this position was already assigned an ID
            if fruit_key in self.detected_fruits:
                current_id = self.detected_fruits[fruit_key]
            elif len(self.detected_fruits) < 4:
                # Assign new ID
                current_id = len(self.detected_fruits) + 1
                self.detected_fruits[fruit_key] = current_id
                self.get_logger().info(
                    f"✅ BAD FRUIT {current_id} DETECTED: [{X:.3f}, {Y:.3f}, {Z:.3f}]"
                )
            else:
                # Already have 4 fruits, skip additional detections
                continue

            # Store for continuous publishing
            self.last_fruit_positions[fruit_key] = (X, Y, Z, current_id)

            # Publish TF
            frame_id = f"{TEAM_ID}_bad_fruit_{current_id}"
            self.publish_tf(frame_id, X, Y, Z)

            # Draw annotation
            cv2.circle(image_disp, (cx_full, cy_full), 10, (0, 0, 255), -1)
            cv2.putText(image_disp, f"bad_fruit_{current_id}", (cx_full - 40, cy_full - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.drawContours(image_disp, [cnt + np.array([roi_x1, roi_y1])], -1, (0, 255, 0), 2)

        return image_disp

    def color_callback(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Color callback error: {e}")
            return

        if self.depth_image is None:
            return

        self.frame_count += 1

        # Work on a copy for display
        image_disp = self.color_image.copy()

        # Detect fertilizer can using ArUco
        image_disp = self.detect_fertilizer_aruco(image_disp)

        # Detect bad fruits
        image_disp = self.detect_bad_fruits(image_disp)

        # Status overlay
        status_text = f"Fruits: {len(self.detected_fruits)}/4 | Fertilizer: {'YES' if self.last_fertilizer_position else 'NO'}"
        cv2.putText(image_disp, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Mode indicator
        mode_text = "HARDWARE" if USE_HARDWARE else "SIMULATION"
        cv2.putText(image_disp, mode_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Periodic status log
        if self.frame_count % 200 == 0:
            self.get_logger().info(
                f"📊 Status: {len(self.detected_fruits)}/4 fruits, "
                f"Fertilizer: {'✓' if self.last_fertilizer_position else '✗'}"
            )

        cv2.imshow("Enhanced Detection - Task 3A", image_disp)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = EnhancedDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Shutting down detector...')
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
