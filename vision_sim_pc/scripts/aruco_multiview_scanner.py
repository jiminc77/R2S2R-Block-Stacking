#!/usr/bin/env python3

import os
import time
import json
import threading
from collections import defaultdict

import numpy as np
from cv2 import aruco
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

import rospy
import actionlib
import tf2_ros
from geometry_msgs.msg import PoseArray, Pose
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

# ============================================================================
# Configuration
# ============================================================================
class ScanConfig:
    """Configuration for Scanning Process"""
    SCAN_RATE_HZ = 15.0              # Detection loop frequency
    MIN_DETECTIONS = 5               # Minimum detections to consider a block valid
    OUTLIER_THRESHOLD_M = 0.015      # Distance threshold for outlier rejection (20mm)
    CONFIDENCE_STD_SCALE = 0.003     # Scale for confidence score calculation (4mm)
    
    MARKER_SIZE_M = 0.040            # ArUco marker size
    STABILIZE_DELAY_S = 1.5          # Wait time after robot motion
    SCAN_DURATION_S = 3.0            # Duration for data collection per view
    
    # Z-Plane Constraint Settings
    # Stacked blocks are at Z + n * Height.
    FIXED_BLOCK_Z = 0.097            # Base layer height (m)
    BLOCK_LAYER_HEIGHT_M = 0.060     # Block height (60mm)

# ============================================================================
# Main Scanner Class
# ============================================================================
class ArucoMultiViewScanner:
    """
    ArUco Marker-based Multi-View Scanner
    
    Moves the robot along a sequence of viewpoints, fuses detections from 
    multiple angles, and applies dense depth plane fitting for precision.
    """

    def __init__(self):
        rospy.init_node('aruco_scanner', anonymous=True)
        
        # Data Buffer: {marker_id: [detection_data, ...]}
        self.detection_buffer = defaultdict(list)
        self.buffer_lock = threading.Lock()
        
        # State Flags
        self.is_scanning = False
        self.data_collection_enabled = False
        
        self.current_view_name = "NONE"
        
        # Initialize Components
        self._init_action_client()
        self._init_tf()
        self._load_geometry_offset()
        self._init_camera()
        self._init_detector()
        self._init_publisher()
        
        rospy.loginfo("✅ ArucoMultiViewScanner Initialized")

    # ------------------------------------------------------------------------
    # 1. Initialization
    # ------------------------------------------------------------------------
    def _init_action_client(self):
        """Connect to Robot Motion Action Server"""
        self.client = actionlib.SimpleActionClient('robot_motion', MoveBaseAction)
        rospy.loginfo("Waiting for 'robot_motion' server...")
        
        if not self.client.wait_for_server(timeout=rospy.Duration(20.0)):
            rospy.logerr("Action server not available.")
        else:
            rospy.loginfo("✅ Connected to Action Server")

    def _init_tf(self):
        """Initialize TF Listener"""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.T_tool0_cam = None  # Lazy loading

    def _load_geometry_offset(self):
        """Load offset from marker surface to block center"""
        path = os.path.join(os.path.dirname(__file__), "../calibration/geometry_offset.json")
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            # JSON format: [x, y, z] -> Z를 음수로 변환하여 안쪽으로 적용
            raw = np.array(data.get("offset_xyz_m", [0, 0, 0.03]))
            self.offset = np.array([raw[0], raw[1], -abs(raw[2])])
            rospy.loginfo(f"Variable Offset: {self.offset}")
        except Exception:
            self.offset = np.array([0, 0, -0.03])
            rospy.logwarn("Using default offset: [0, 0, -0.03]")

    def _init_camera(self):
        """Start RealSense Camera Stream"""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            profile = self.pipeline.start(config)
            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            
            self.camera_matrix = np.array([
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1.0]
            ])
            self.dist_coeffs = np.array(intrinsics.coeffs)
            
            # Auto-exposure stabilization
            for _ in range(30):
                self.pipeline.wait_for_frames()

            self.temporal_filter = rs.temporal_filter()
            # Align object for depth-color synchronization
            self.align = rs.align(rs.stream.color)
            
            rospy.loginfo(f"✅ Camera Ready (fx={intrinsics.fx:.1f})")
        except Exception as e:
            rospy.logerr(f"Camera Init Failed: {e}")
            self.pipeline = None

    def _init_detector(self):
        """Initialize ArUco Detector Parameters"""
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters_create()
        self.aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 30
        self.aruco_params.cornerRefinementMinAccuracy = 0.01
        self.marker_size = ScanConfig.MARKER_SIZE_M

    def _init_publisher(self):
        """Initialize ROS Publisher"""
        self.block_pose_pub = rospy.Publisher(
            '/block_poses', PoseArray, queue_size=1, latch=True
        )
        rospy.loginfo("✅ Publisher initialized: /block_poses")

    # ------------------------------------------------------------------------
    # 2. Continuous Scanning Logic (Background Thread)
    # ------------------------------------------------------------------------
    def _start_continuous_scan(self):
        self.is_scanning = True
        self.scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.scan_thread.start()

    def _stop_continuous_scan(self):
        self.is_scanning = False
        if hasattr(self, 'scan_thread'):
            self.scan_thread.join(timeout=2.0)

    def _scan_loop(self):
        """Continuous background scanning loop"""
        rate = rospy.Rate(ScanConfig.SCAN_RATE_HZ)
        
        while self.is_scanning and not rospy.is_shutdown():
            try:
                if self.pipeline:
                    frames = self.pipeline.wait_for_frames(2000)
                    aligned_frames = self.align.process(frames)
                    
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    
                    if color_frame and depth_frame:
                        depth_frame_filtered = self.temporal_filter.process(depth_frame).as_depth_frame()
                        
                        if self.data_collection_enabled:
                            detections = self._detect_markers(color_frame, depth_frame_filtered)
                            if detections:
                                self._process_and_buffer(detections)
                                
            except Exception as e:
                rospy.logwarn(f"Scan Loop Warning: {e}")

            rate.sleep()

    def _detect_markers(self, color_frame, depth_frame):
        """Estimate poses using provided frames"""
        try:
            img = np.asanyarray(color_frame.get_data())
            corners, ids, _ = aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_params)

            results = {}
            if ids is not None:
                intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
                
                for i, marker_id in enumerate(ids.flatten()):
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                        corners[i:i+1], self.marker_size, self.camera_matrix, self.dist_coeffs
                    )
                    rvec = rvec[0][0]
                    tvec = tvec[0][0]
                    
                    c_x = int(np.mean(corners[i][0][:, 0]))
                    c_y = int(np.mean(corners[i][0][:, 1]))
                    
                    min_x = int(np.min(corners[i][0][:, 0]))
                    max_x = int(np.max(corners[i][0][:, 0]))
                    min_y = int(np.min(corners[i][0][:, 1]))
                    max_y = int(np.max(corners[i][0][:, 1]))
                     
                    points_3d = []
                    step = 2 
                    for y in range(min_y, max_y, step):
                        for x in range(min_x, max_x, step):
                             if 0 <= x < 640 and 0 <= y < 480:
                                 # 이미 필터링된 depth_frame을 사용
                                 d = depth_frame.get_distance(x, y)
                                 if 0.1 < d < 2.0 and abs(d - np.linalg.norm(tvec)) < 0.1:
                                     p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                                     points_3d.append(p)
                    
                    final_pos = tvec
                    final_rvec = rvec
                    
                    if len(points_3d) > 20:
                        points_3d = np.array(points_3d)
                        normal, plane_centroid = self._fit_plane_ransac(points_3d)
                        
                        if normal is not None:
                            center_ray = rs.rs2_deproject_pixel_to_point(intrinsics, [c_x, c_y], 1.0)
                            center_ray = np.array(center_ray) / np.linalg.norm(center_ray)
                            
                            denom = np.dot(center_ray, normal)
                            if abs(denom) > 1e-6:
                                t = np.dot(plane_centroid, normal) / denom
                                final_pos = center_ray * t
                            
                            rmat_aruco = R.from_rotvec(rvec).as_dcm()
                            z_axis_aruco = rmat_aruco[:, 2]
                            
                            if np.dot(normal, z_axis_aruco) < 0:
                                normal = -normal
                                
                            cross = np.cross(z_axis_aruco, normal)
                            sine = np.linalg.norm(cross)
                            cosine = np.dot(z_axis_aruco, normal)
                            
                            if sine > 1e-6:
                                axis = cross / sine
                                angle = np.arctan2(sine, cosine)
                                r_delta = R.from_rotvec(axis * angle)
                                rmat_refined = r_delta.as_dcm() @ rmat_aruco
                                final_rvec = R.from_dcm(rmat_refined).as_rotvec()

                    results[marker_id] = {'rvec': final_rvec, 'tvec': final_pos}
            
            return results

        except Exception as e:
            rospy.logerr_throttle(1.0, f"Detection Error: {e}") 
            return {}

    def _process_and_buffer(self, raw_detections):
        """Transform poses to Base Frame and buffer them"""
        # TF 조회 (Base -> Tool, Tool -> Camera)
        try:
            if self.T_tool0_cam is None:
                self.T_tool0_cam = self._get_tf_matrix("tool0", "camera_color_optical_frame")
            T_base_tool = self._get_tf_matrix("base_link", "tool0")
        except Exception:
            return

        if T_base_tool is None or self.T_tool0_cam is None: return

        # Pre-calculate Base -> Camera transform
        T_base_cam = T_base_tool @ self.T_tool0_cam
        cam_pos_base = T_base_cam[:3, 3]

        with self.buffer_lock:
            for marker_id, data in raw_detections.items():
                # 1. Camera Frame -> Marker Frame
                T_cam_marker = np.eye(4)
                T_cam_marker[:3, :3] = R.from_rotvec(data['rvec']).as_dcm()
                T_cam_marker[:3, 3] = data['tvec']
                
                # 2. Base Frame -> Marker Frame
                T_base_marker = T_base_cam @ T_cam_marker
                
                marker_pos = T_base_marker[:3, 3]
                marker_rot_obj = R.from_dcm(T_base_marker[:3, :3])

                rot_correction = R.from_euler('x', -90, degrees=True)
                block_rot_obj = marker_rot_obj * rot_correction
                
                # 3. Apply Offset (Marker Surface -> Block Center)
                offset_world = marker_rot_obj.apply(self.offset)
                block_pos_noisy = marker_pos + offset_world
                
                # 4. Store
                self.detection_buffer[marker_id].append({
                    'block_pos_noisy': block_pos_noisy, # Raw 3D pos (with noisy Z)
                    'cam_pos': cam_pos_base,            # Camera position for ray reconstruction
                    'block_rot': block_rot_obj,
                    'view_name': self.current_view_name
                })

    def _get_tf_matrix(self, target, source):
        """Get 4x4 transform matrix from TF"""
        try:
            t = self.tf_buffer.lookup_transform(target, source, rospy.Time(0), rospy.Duration(0.5))
            mat = np.eye(4)
            mat[:3, :3] = R.from_quat([t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]).as_dcm()
            mat[:3, 3] = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
            return mat
        except Exception:
            return None

    # ------------------------------------------------------------------------
    # 3. Motion Control
    # ------------------------------------------------------------------------
    def move_robot(self, target_name):
        """Send navigation goal to Action Server"""
        rospy.loginfo(f"Target: {target_name}")
        
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = target_name
        goal.target_pose.header.stamp = rospy.Time.now()
        
        self.client.send_goal(goal)
        finished = self.client.wait_for_result(rospy.Duration(60.0))
        
        if not finished or self.client.get_state() != actionlib.GoalStatus.SUCCEEDED:
            rospy.logwarn(f"Failed to reach {target_name}")
            return False
        rospy.loginfo(f"Reached {target_name}")
        return True

    # ------------------------------------------------------------------------
    # 4. Data Processing (Fusion + Global Optimization)
    # ------------------------------------------------------------------------
    def _calculate_results(self):
        """Fuse buffered data and apply global optimization"""
        with self.buffer_lock:
            raw_data = dict(self.detection_buffer)
            
        initial_estimates = {}
        
        # 1. Individual Fusion (Weighted Average)
        grouped = defaultdict(list)
        for mid, records in raw_data.items():
            if len(records) < ScanConfig.MIN_DETECTIONS: 
                rospy.logwarn(f"Marker {mid} ignored (Count {len(records)} < {ScanConfig.MIN_DETECTIONS})")
                continue
            block_id = mid // 4
            grouped[block_id].extend(records)
            
        for bid, records in grouped.items():
            result = self._fuse_block_data(records, bid)
            if result:
                initial_estimates[bid] = result

        if not initial_estimates:
            return {}

        # 2. Global Layer Optimization (Relative Snapping)
        # User Constraint: Table Height = 17cm, Block Center (Layer 0) = 20cm
        base_layer_z = ScanConfig.FIXED_BLOCK_Z
        
        rospy.loginfo(f"\n[Global Optimization] Enforcing Base Layer Z: {base_layer_z:.4f} m")
        
        final_blocks = {}
        for bid, data in initial_estimates.items():
            current_z = data['position'][2]
            
            # Determine Layer Index: Round((Z - Base) / Height)
            layer_idx = round((current_z - base_layer_z) / ScanConfig.BLOCK_LAYER_HEIGHT_M)
            layer_idx = max(0, layer_idx)
            
            # Snap Z: Base + n * Height
            snapped_z = base_layer_z + (layer_idx * ScanConfig.BLOCK_LAYER_HEIGHT_M)
            
            rospy.loginfo(f"  > Block {bid}: Raw Z={current_z:.3f} -> Layer {layer_idx} (Z={snapped_z:.3f})")
            rospy.loginfo(f"  [DEBUG] Raw Z before snapping: {current_z:.4f} m (Target Base: {base_layer_z:.4f} m)")
            
            data['position'][2] = snapped_z
            final_blocks[bid] = data
            
        return final_blocks

    def _fuse_block_data(self, records, block_id):
        """Compute robust weighted average for block pose"""
        # We use the raw 'block_pos_noisy' which is simple 3D triangulation from ArUco
        positions = np.array([r['block_pos_noisy'] for r in records])
        rotations = [r['block_rot'] for r in records]
        
        # 1. Outlier Removal (3D Euclidean)
        median = np.median(positions, axis=0)
        dists = np.linalg.norm(positions - median, axis=1)
        mask = dists <= ScanConfig.OUTLIER_THRESHOLD_M
        
        valid_pos = positions[mask]
        valid_rot = [rot for rot, m in zip(rotations, mask) if m]
        
        if len(valid_pos) == 0: return None
        
        # 2. Weighted Average for Position (Use ALL valid inliers)
        # Weighting Strategy:
        # - Inverse Distance (Closer is better)
        # - View Priority (CENTER view is best for X/Y)
        
        weights = []
        for i, pos in enumerate(valid_pos):
             dist = dists[mask][i]
             w = 1.0 / (dist + 1e-6)
             weights.append(w)

             
        # Re-implementation of Weighting with View Name
        valid_records = [r for r, m in zip(records, mask) if m]
        valid_pos_f = np.array([r['block_pos_noisy'] for r in valid_records])
        
        final_weights = []
        for i, r in enumerate(valid_records):
            # Base weight from distance (tight cluster)
            d = np.linalg.norm(valid_pos_f[i] - median)
            w = 1.0 / (d + 1e-6)
            
            # View Boost
            if r['view_name'] == "CENTER":
                w *= 2.0 # Double confidence for Top-Down view
            
            final_weights.append(w)
            
        avg_pos = np.average(valid_pos_f, axis=0, weights=final_weights)
        
        # 3. Rotation Strategy: Clustering & Averaging
        # Filter records by the valid mask first to ensure we use inlier data
        valid_records = [r for r, m in zip(records, mask) if m]
        
        quats = np.array([r.as_quat() for r in valid_rot])
        if len(quats) > 0:
            # Simple greedy clustering
            clusters = []
            unused = set(range(len(quats)))
            
            while unused:
                seed_idx = next(iter(unused))
                seed_q = quats[seed_idx]
                
                cluster = [seed_idx]
                unused.remove(seed_idx)
                
                # Verify others against seed
                to_remove = []
                for idx in unused:
                    # Dot product > 0.707 means < 90 degrees diff approx
                    if abs(np.dot(seed_q, quats[idx])) > 0.8: # ~70 deg threshold
                        cluster.append(idx)
                        to_remove.append(idx)
                
                for idx in to_remove:
                    unused.remove(idx)
                
                clusters.append(cluster)
            
            # Pick largest cluster
            largest_cluster_indices = max(clusters, key=len)
            dominant_rotations = [valid_rot[i] for i in largest_cluster_indices]
            avg_rot = self._average_quaternions(dominant_rotations)
            
            n_rot_inliers = len(dominant_rotations)
        else:
             avg_rot = self._average_quaternions(valid_rot)
             n_rot_inliers = 0

        yaw, pitch, roll = avg_rot.as_euler('zyx', degrees=False)
        avg_rot = R.from_euler('zyx', [yaw, 0, 0], degrees=False)

        # 4. Statistics
        std_vec = np.std(valid_pos, axis=0)
        avg_std = std_vec.mean() # 3D Jitter
        
        ratio_score = len(valid_pos) / len(positions)
        precision_score = np.exp(-avg_std / ScanConfig.CONFIDENCE_STD_SCALE)
        # Penalty if dominant rotation cluster is small compared to valid positions
        rot_consistency = n_rot_inliers / len(valid_pos) if len(valid_pos) > 0 else 0
        
        conf = ratio_score * precision_score * rot_consistency
        
        rospy.loginfo(f"\n[Block Fusion Weighted] Block ID: {block_id}")
        rospy.loginfo(f"  > Samples: {len(positions)} -> Inliers: {len(valid_pos)}")
        rospy.loginfo(f"  > 3D Jitter (mm): X={std_vec[0]*1000:.2f}, Y={std_vec[1]*1000:.2f}, Z={std_vec[2]*1000:.2f} (Avg={avg_std*1000:.2f})")
        rospy.loginfo(f"  > Confidence: {conf:.2f}")

        return {
            'position': avg_pos,
            'rotation': avg_rot,
            'confidence': conf,
            'count': len(valid_pos)
        }

    def _average_quaternions(self, rotations):
        """Averages a list of rotations"""
        quats = np.array([r.as_quat() for r in rotations])
        # Antipodal handling
        for i in range(1, len(quats)):
            if np.dot(quats[0], quats[i]) < 0:
                quats[i] = -quats[i]
        avg = np.mean(quats, axis=0)
        return R.from_quat(avg / np.linalg.norm(avg))

    # ------------------------------------------------------------------------
    # 5.1 Geometry Helpers (RANSAC)
    # ------------------------------------------------------------------------
    def _fit_plane_ransac(self, points, n_iter=100, threshold=0.0015):
        """
        Fits a plane to 3D points using RANSAC.
        Returns (normal, centroid)
        """
        n_points = len(points)
        if n_points < 3: return None, None
        
        best_inliers = []
        best_plane = None
        
        # Optimize: vectorized random sampling could be faster but loop is fine for N~200
        for _ in range(n_iter):
            # Sample 3 random points
            try:
                sample_indices = np.random.choice(n_points, 3, replace=False)
            except ValueError:
                 return None, None # Not enough points?
                 
            p1, p2, p3 = points[sample_indices]
            
            # Compute Plane Normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6: continue # Collinear
            normal /= norm
            
            # Distance from point to plane: |(p - p1) . n|
            # Vectorized distance check
            # Plane eq: n . (x - p1) = 0
            dists = np.abs(np.dot(points - p1, normal))
            inliers = np.where(dists < threshold)[0]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (normal, p1) # Using p1 as temporary centroid
        
        if len(best_inliers) < 10:
             return None, None
             
        # Refine with all inliers using SVD
        inlier_points = points[best_inliers]
        centroid = np.mean(inlier_points, axis=0)
        centered = inlier_points - centroid
        
        # SVD
        u, s, vh = np.linalg.svd(centered)
        # Normal is the last row of vh (smallest singular value direction)
        normal = vh[2, :]
        
        return normal, centroid


    # ------------------------------------------------------------------------
    # 5. Publishing
    # ------------------------------------------------------------------------
    def _publish_results(self, fused_blocks):
        """Publish fused results to /block_poses topic"""
        msg = PoseArray()
        msg.header.frame_id = "base_link"
        msg.header.stamp = rospy.Time.now()
        
        for block_id in sorted(fused_blocks.keys()):
            data = fused_blocks[block_id]
            
            pose = Pose()
            pose.position.x = data['position'][0]
            pose.position.y = data['position'][1]
            pose.position.z = data['position'][2]
            
            quat = data['rotation'].as_quat()
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            
            msg.poses.append(pose)
        
        self.block_pose_pub.publish(msg)
        rospy.loginfo(f"📡 Published {len(msg.poses)} block poses to /block_poses")

    # ------------------------------------------------------------------------
    # 6. Main Sequence
    # ------------------------------------------------------------------------
    def execute_scan(self):
        """Execute the full scanning sequence"""
        # (TargetName, ScanEnabled, Duration)
        SEQUENCE = [
            ("HOME",   False, 1.0),
            ("RIGHT",  True,  3.0),
            ("CENTER", True,  3.0),
            ("LEFT",   True,  3.0),
            ("CENTER", True,  3.0),
            ("HOME",   False, 0.0)
        ]
        
        rospy.loginfo("--- Starting Scan Sequence ---")
        with self.buffer_lock: self.detection_buffer.clear()
        
        self._start_continuous_scan()
        
        try:
            for i, (target, scan, wait) in enumerate(SEQUENCE):
                rospy.loginfo(f"[{i+1}/{len(SEQUENCE)}] Moving to {target}...")
                
                self.current_view_name = target # Set Current View Name
                self.data_collection_enabled = False
                if not self.move_robot(target):
                    continue
                if scan:
                    time.sleep(ScanConfig.STABILIZE_DELAY_S)
                    self.data_collection_enabled = True
                    rospy.loginfo(f"   📸 Scanning ({ScanConfig.SCAN_DURATION_S}s)...")
                    time.sleep(ScanConfig.SCAN_DURATION_S)
                    self.data_collection_enabled = False
                elif wait > 0:
                    time.sleep(wait)
        finally:
            self._stop_continuous_scan()
            
            
        rospy.loginfo("--- Scan Complete. Calculating Results ---")
        final_results = self._calculate_results()
                    
        return final_results

# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    try:
        # 1. Get User Input
        try:
            input_val = input("Enter expected number of blocks: ")
            expected_count = int(input_val)
        except ValueError:
            print("Invalid input. Defaulting to 0 (No check).")
            expected_count = 0

        scanner = ArucoMultiViewScanner()
        
        # 2. Retry Logic (Max 3 attempts)
        max_attempts = 3
        results = {}
        
        for attempt in range(1, max_attempts + 1):
            rospy.loginfo(f"🚀 Starting Scan Attempt {attempt}/{max_attempts}")
            
            results = scanner.execute_scan()
            detected_count = len(results)
            
            print("\n" + "="*50)
            print(f" ATTEMPT {attempt} RESULT: Found {detected_count} blocks (Expected: {expected_count})")
            print("="*50)
            
            if expected_count > 0 and detected_count != expected_count:
                if attempt < max_attempts:
                    rospy.logwarn(f"❌ Mismatch detected! (Found {detected_count}, Expected {expected_count}). Retrying in 3 seconds...")
                    time.sleep(3.0)
                    continue
                else:
                    rospy.logerr(f"❌ Max attempts reached. Proceeding with {detected_count} blocks.")
                    if results: scanner._publish_results(results)  # Publish best effort
            else:
                 rospy.loginfo("✅ Block count matches (or no check). Success!")
                 if results: scanner._publish_results(results)  # Publish success
                 break

        print("\n" + "="*50)
        print(f" FINAL RESULTS (Found {len(results)} blocks)")
        print("="*50)
        
        for bid, data in sorted(results.items()):
            p = data['position']
            e = data['rotation'].as_euler('xyz', degrees=True)
            print(f"📦 Block {bid}:")
            print(f"   Pos: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] m")
            print(f"   Rot: [{e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}] deg")
            print(f"   Conf: {data['confidence']:.2f}")
            print("-" * 30)

        rospy.loginfo("Scan complete. Node is keeping alive to publish topic (latch=True).")
        rospy.loginfo("Scanning finished. Node spinning...")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass
