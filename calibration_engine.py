"""
KnightAngle ADAS Calibration Platform
Core Mathematical Engine: Dual-IMU Compensation & Inverse Homography Warping

Implements:
1. Dual-IMU Error Separation: Isolates Error B1 (human mounting error) from Error A (vehicle floor slope).
2. Virtual Camera Rotation: Cancels Error B1 using inverse rotation matrix R_{B1}^-1.
3. Homography Computation: Calculates perspective warp matrix H_A^-1 to counter floor incline.
4. Target Warper: Synthesizes geometrically corrected calibration patterns for digital projection.
"""

import numpy as np
import cv2


class DualIMUCompensator:
    """
    Handles IMU sensor fusion and mathematical error isolation.
    IMU-1: Vehicle Gravity Datum (Floor slope / Incline Error A)
    IMU-2: Reference Module Chassis (Error A + Human Mounting Error B1)
    """
    def __init__(self, fixed_mounting_error=None):
        # Default fixed human mounting error (pitch, roll, yaw in degrees)
        # e.g., technician mounted camera slightly tilted by +1.5 deg pitch, -0.8 deg roll
        if fixed_mounting_error is None:
            self.b1_deg = np.array([1.5, -0.8, 0.0], dtype=np.float64)
        else:
            self.b1_deg = np.array(fixed_mounting_error, dtype=np.float64)

    @staticmethod
    def euler_to_rotation_matrix(pitch_deg, roll_deg, yaw_deg):
        """
        Converts Euler angles (degrees) to 3x3 rotation matrix (XYZ convention).
        """
        p = np.radians(pitch_deg)
        r = np.radians(roll_deg)
        y = np.radians(yaw_deg)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(p), -np.sin(p)],
            [0, np.sin(p), np.cos(p)]
        ], dtype=np.float64)

        Ry = np.array([
            [np.cos(r), 0, np.sin(r)],
            [0, 1, 0],
            [-np.sin(r), 0, np.cos(r)]
        ], dtype=np.float64)

        Rz = np.array([
            [np.cos(y), -np.sin(y), 0],
            [np.sin(y), np.cos(y), 0],
            [0, 0, 1]
        ], dtype=np.float64)

        return Rz @ Ry @ Rx

    @staticmethod
    def rotation_matrix_to_euler(R):
        """
        Converts 3x3 rotation matrix back to Euler angles (degrees).
        """
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            roll = np.degrees(np.arctan2(-R[2, 0], sy))
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
            roll = np.degrees(np.arctan2(-R[2, 0], sy))
            yaw = 0.0

        return np.array([pitch, roll, yaw], dtype=np.float64)

    def compute_mounting_error(self, imu1_deg, imu2_deg):
        """
        Subtracts IMU1 from IMU2 in SO(3) rotation space to isolate pure Error B1.
        R_B1 = R_IMU1^T * R_IMU2
        """
        R_imu1 = self.euler_to_rotation_matrix(*imu1_deg)
        R_imu2 = self.euler_to_rotation_matrix(*imu2_deg)

        # Relative rotation: R_B1 = R_imu1.T @ R_imu2
        R_B1 = R_imu1.T @ R_imu2
        isolated_b1_deg = self.rotation_matrix_to_euler(R_B1)
        return isolated_b1_deg, R_B1

    def compute_virtual_rotation(self, R_ref_camera, R_B1):
        """
        Applies inverse mounting rotation R_B1^-1 to Reference Camera.
        R_virtual = R_ref_camera * R_B1^-1
        The virtual camera is now perfectly aligned with the vehicle body datum!
        """
        R_B1_inv = R_B1.T
        R_virtual = R_ref_camera @ R_B1_inv
        return R_virtual

    def rectify_camera_stream(self, raw_frame, R_B1, focal_length=1500.0):
        """
        Applies virtual camera rotation R_B1^-1 to rectify the Reference Camera feed.
        H_rect = K * R_B1^-1 * K^-1
        The resulting stream represents what an ideally mounted reference camera
        (perfectly parallel to vehicle body datum) perceives!
        """
        h, w = raw_frame.shape[:2]
        K = np.array([
            [focal_length, 0.0, w / 2.0],
            [0.0, focal_length, h / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        R_B1_inv = R_B1.T
        H_rect = K @ R_B1_inv @ np.linalg.inv(K)
        if abs(H_rect[2, 2]) > 1e-9:
            H_rect /= H_rect[2, 2]

        rectified = cv2.warpPerspective(raw_frame, H_rect, (w, h), borderValue=(15, 15, 20))
        return rectified, H_rect


def estimate_error_a_from_corners(corners):
    """
    Computes floor/environmental tilt (Error A) from ChArUco marker geometry
    observed by the datum-rectified Reference Camera:
    - Pitch: Estimated from vertical keystone (top width vs bottom width)
    - Roll: Estimated from in-plane tilt angle of horizontal edges
    """
    if corners is None or len(corners) == 0:
        return 0.0, 0.0
    geom = measure_target_geometry(corners)
    if geom is None:
        return 0.0, 0.0

    measured_roll = geom['tilt_deg']
    keystone = geom['keystone_pct']
    sign = 1.0 if geom['width_ratio'] >= 1.0 else -1.0
    measured_pitch = sign * (keystone / 1.0)

    return float(measured_pitch), float(measured_roll)


class HomographyWarpEngine:
    """
    Computes inverse homography to warp calibration patterns for digital screen projection.
    Compensates for vehicle tilt/pitch/roll (Error A) relative to the screen plane.
    """
    def __init__(self, target_width=1200, target_height=720, distance_meters=2.0, focal_length=None):
        self.width = target_width
        self.height = target_height
        self.distance = distance_meters

        if focal_length is None:
            # Subtended half-angle at distance: tan(theta/2) = (screen_width / 2) / distance
            # For 2.8m screen at distance_meters:
            screen_width = 2.80 if distance_meters > 2.5 else 1.60
            half_tan = max(1e-4, (screen_width / 2.0) / max(0.5, distance_meters))
            focal_length = (target_width / 2.0) / half_tan

        self.focal_length = focal_length

        # Camera intrinsic matrix K
        self.K = np.array([
            [focal_length, 0.0, target_width / 2.0],
            [0.0, focal_length, target_height / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)

    def compute_inverse_homography(self, pitch_deg, roll_deg, yaw_deg=0.0):
        """
        Calculates exact projective homography matrix H = K * R_comp * K^-1
        Pre-warps the screen pattern in the exact OPPOSITE direction of the vehicle's
        floor slope / tilt, so that the tilted ADAS camera perceives a perfectly square,
        rectilinear target.
        """
        # In camera coordinates (X right, Y down, Z forward along optical axis):
        # - Rx(p): Pitch compensation (up/down vehicle tilt -> vertical keystone)
        # - Ry(y): Yaw compensation (heading error -> horizontal keystone)
        # - Rz(r): Roll compensation (lateral vehicle lean -> in-plane 2D rotation)
        p = np.radians(pitch_deg)
        r = np.radians(roll_deg)
        y = np.radians(yaw_deg)

        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(p), -np.sin(p)],
            [0.0, np.sin(p), np.cos(p)]
        ], dtype=np.float64)

        Ry = np.array([
            [np.cos(y), 0.0, np.sin(y)],
            [0.0, 1.0, 0.0],
            [-np.sin(y), 0.0, np.cos(y)]
        ], dtype=np.float64)

        Rz = np.array([
            [np.cos(r), -np.sin(r), 0.0],
            [np.sin(r), np.cos(r), 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        R_comp = Rz @ Ry @ Rx

        # Planar rotation-induced homography
        H = self.K @ R_comp @ self.K_inv

        # Centering constraint: preserves the target centroid on the physical display screen,
        # preventing the warped pattern from drifting off-screen under pitch and roll.
        c = np.array([self.width / 2.0, self.height / 2.0, 1.0])
        c_prime = H @ c
        if abs(c_prime[2]) > 1e-9:
            c_prime /= c_prime[2]
            dx = (self.width / 2.0) - c_prime[0]
            dy = (self.height / 2.0) - c_prime[1]
            T_center = np.array([
                [1.0, 0.0, dx],
                [0.0, 1.0, dy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            H = T_center @ H

        # Normalize so bottom-right element is 1.0
        if abs(H[2, 2]) > 1e-9:
            H = H / H[2, 2]

        return H

    def warp_target(self, target_image, pitch_deg, roll_deg, yaw_deg=0.0):
        """
        Warps the target image using inverse perspective homography.
        """
        H = self.compute_inverse_homography(pitch_deg, roll_deg, yaw_deg)
        h, w = target_image.shape[:2]
        warped = cv2.warpPerspective(target_image, H, (w, h), borderValue=(255, 255, 255))
        return warped, H


def measure_target_geometry(corners):
    """
    Computes rigorous empirical geometric distortion metrics from detected marker corners:
    - Width Ratio: W_top / W_bot (measures trapezoidal keystoning; 1.000 = perfectly rectilinear)
    - Keystone Distortion: |1.0 - (W_top / W_bot)| * 100%
    - Measured In-Plane Tilt: Angular rotation relative to camera horizon in degrees
    - Height Ratio: H_left / H_right (measures lateral skew)
    """
    if corners is None or len(corners) == 0:
        return None
    pts = np.vstack([c.reshape(-1, 2) for c in corners])
    if len(pts) < 4:
        return None

    hull = cv2.convexHull(pts.astype(np.float32))
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
    if len(approx) == 4:
        quad_pts = approx.reshape(-1, 2)
    else:
        rect = cv2.minAreaRect(pts)
        quad_pts = cv2.boxPoints(rect)

    center = np.mean(quad_pts, axis=0)

    # Sort corners clockwise starting from top-left using polar angle around center
    angles = np.arctan2(quad_pts[:, 1] - center[1], quad_pts[:, 0] - center[0])
    sorted_idx = np.argsort(angles)
    tl = quad_pts[sorted_idx[0]]
    tr = quad_pts[sorted_idx[1]]
    br = quad_pts[sorted_idx[2]]
    bl = quad_pts[sorted_idx[3]]

    w_top = float(np.linalg.norm(tr - tl))
    w_bot = float(np.linalg.norm(br - bl))
    h_l = float(np.linalg.norm(bl - tl))
    h_r = float(np.linalg.norm(br - tr))

    w_ratio = (w_top / w_bot) if w_bot > 1e-3 else 1.0
    keystone_pct = abs(1.0 - w_ratio) * 100.0

    angle_top = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
    angle_bot = np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))
    tilt_deg = float(0.5 * (angle_top + angle_bot))

    return {
        'w_top': w_top,
        'w_bot': w_bot,
        'width_ratio': w_ratio,
        'keystone_pct': keystone_pct,
        'tilt_deg': tilt_deg,
        'h_ratio': (h_l / h_r) if h_r > 1e-3 else 1.0,
        'quad': np.array([tl, tr, br, bl], dtype=np.int32)
    }


def calibrate_adas_camera(corners):
    """
    Computes pure Car ADAS Camera mounting error (B_ADAS) from the pre-compensated
    calibration target. Because environmental Error A is neutralized by the pre-warped screen,
    the measured residual keystone and in-plane tilt directly represent B_ADAS.
    """
    geom = measure_target_geometry(corners)
    if geom is None:
        return {'b_adas_pitch': 0.0, 'b_adas_roll': 0.0, 'status': 'NO_TARGET', 'calibrated': False, 'quad': None}

    # In-plane tilt = roll mounting error
    b_adas_roll = geom['tilt_deg']
    # Keystone ratio = pitch mounting error
    keystone = geom['keystone_pct']
    sign = 1.0 if geom['width_ratio'] >= 1.0 else -1.0
    b_adas_pitch = sign * (keystone / 1.0)

    is_calibrated = (abs(b_adas_pitch) < 0.3 and abs(b_adas_roll) < 0.3)
    return {
        'b_adas_pitch': float(b_adas_pitch),
        'b_adas_roll': float(b_adas_roll),
        'keystone_pct': geom['keystone_pct'],
        'width_ratio': geom['width_ratio'],
        'tilt_deg': geom['tilt_deg'],
        'status': 'CALIBRATION PASS' if is_calibrated else 'MEASURING B_ADAS',
        'calibrated': is_calibrated,
        'quad': geom['quad']
    }




def get_or_create_charuco_target(image_path="ChArUco.jpg", width=1200, height=720):
    """
    Loads ChArUco.jpg if present, or dynamically synthesizes an authentic 9x6 ChArUco
    calibration board covering the full digital display screen.
    """
    import os
    if os.path.exists(image_path):
        loaded = cv2.imread(image_path)
        if loaded is not None:
            return cv2.resize(loaded, (width, height))

    # Generate high-density 9x6 ChArUco board
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((9, 6), 0.04, 0.025, dict_4x4)
    charuco_img = board.generateImage((width, height), marginSize=35)
    target = cv2.cvtColor(charuco_img, cv2.COLOR_GRAY2BGR)

    # Add professional OEM datum annotations
    cv2.putText(target, "OEM ADAS CALIBRATION DATUM - CHARUCO GRID 9x6", (35, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2)
    cv2.rectangle(target, (2, 2), (width - 2, height - 2), (0, 0, 0), 4)

    # Save to disk for future runs and teammate inspection
    cv2.imwrite(image_path, target)
    print(f"[TargetEngine] Generated and saved authentic calibration board: {image_path}", flush=True)
    return target


# Alias for backward compatibility
generate_oem_charuco_target = get_or_create_charuco_target
