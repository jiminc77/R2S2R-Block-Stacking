import pyrealsense2 as rs
import numpy as np
import cv2
from cv2 import aruco

def test_camera_and_aruco():
    """Test D435 streaming and ArUco detection"""

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    print("Starting RealSense D435...")
    pipeline.start(config)

    # ArUco detector
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    print("✅ Camera streaming started")
    print("Place blocks in view and press 'q' to quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Detect ArUco markers
            corners, ids, rejected = detector.detectMarkers(color_image)

            # Draw markers
            if ids is not None:
                aruco.drawDetectedMarkers(color_image, corners, ids)
                for i, marker_id in enumerate(ids):
                    print(f"Detected Block {marker_id[0]+1} (ID {marker_id[0]})")

            cv2.imshow('D435 - ArUco Detection Test', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("✅ Camera test complete")

if __name__ == "__main__":
    test_camera_and_aruco()