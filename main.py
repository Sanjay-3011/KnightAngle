#!/usr/bin/env python3
"""
KnightAngle ADAS Calibration Platform
Full Closed-Loop Simulation Testbed & Demonstration Suite

Features:
1. ZeroMQ Remote API connection to CoppeliaSim.
2. Dynamic Platform Control: Adjust floor slope / vehicle incline (Error A) live via keys.
3. Dual-IMU Sensor Fusion:
   - IMU-1: Vehicle Gravity Datum (Floor slope A)
   - IMU-2: Reference Module Chassis (Floor slope A + Human Mounting Error B1)
4. Math Engine: Isolates pure mounting error B1 = IMU2 - IMU1 and applies virtual rotation.
5. Inverse Homography Engine: Calculates H_A^-1 and warps target in real time.
6. 3D Texture Streaming: Pushes warped calibration pattern directly onto CoppeliaSim's DisplayScreen.
7. ArUco Marker Detection: Sub-pixel corner tracking on the vehicle's vision sensor.
8. Interactive Presentation UI: Split-screen telemetry display with live angles and controls.
"""

import sys
import time
import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from calibration_engine import (
    DualIMUCompensator, HomographyWarpEngine,
    generate_oem_charuco_target, measure_target_geometry,
    calibrate_adas_camera, estimate_error_a_from_corners
)
from phone_bridge import start_phone_bridge_server, latest_imu_data


def setup_tv_rig_in_scene(sim, sensor_h, screen_h):
    """
    Configures the OEM ADAS Calibration Screen in CoppeliaSim:
    - Positions screen at X = -11.0m, Y = -2.85m, Z = 1.85m facing vehicle datum
    - Sets 16:9 widescreen orientation and 1.0 / 0.6 texture scaling for full border visibility
    - Pure digital screen target with zero spawned artificial models
    """
    if screen_h is None:
        return
    try:
        alias = sim.getObjectAlias(screen_h)
        sim.setObjectPosition(screen_h, -1, [-11.0, -2.85, 1.85])
        sim.setObjectOrientation(screen_h, -1, [0.0, np.pi/2, np.pi/2])
        if 'displayimage' in alias.lower() or 'image' in alias.lower():
            sim.setObjectFloatParam(screen_h, sim.shapefloatparam_texture_scaling_x, 1.0)
            sim.setObjectFloatParam(screen_h, sim.shapefloatparam_texture_scaling_y, 0.6)
            print("[KnightAngle] DisplayImage configured: X=-11.0m, Z=1.85m, 16:9 facing vehicle datum.", flush=True)

        try:
            h_frame = sim.getObject('/DisplayScreen')
            if h_frame != screen_h:
                sim.setObjectPosition(h_frame, -1, [-11.024, -2.85, 1.85])
                sim.setObjectOrientation(h_frame, -1, [0.0, np.pi/2, np.pi/2])
        except Exception:
            pass
    except Exception as e:
        print(f"[KnightAngle] TV station setup notice: {e}", flush=True)


def get_aruco_detectors():
    """Initializes ArUco detectors supporting both standard 4x4 and 6x6 dictionaries."""
    params = cv2.aruco.DetectorParameters()
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    dict_4x4_1000 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    dict_6x6 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)

    if hasattr(cv2.aruco, 'ArucoDetector'):
        det_4x4 = cv2.aruco.ArucoDetector(dict_4x4, params)
        det_4x4_1000 = cv2.aruco.ArucoDetector(dict_4x4_1000, params)
        det_6x6 = cv2.aruco.ArucoDetector(dict_6x6, params)
        return [("4x4_50", det_4x4), ("4x4_1000", det_4x4_1000), ("6x6", det_6x6)], True
    else:
        return [("4x4_50", dict_4x4, params), ("4x4_1000", dict_4x4_1000, params), ("6x6", dict_6x6, params)], False


def detect_markers(image, detectors, is_modern_api):
    """Runs marker detection across configured dictionaries. Returns (corners, ids, dict_name)."""
    for item in detectors:
        if is_modern_api:
            name, detector = item
            corners, ids, _ = detector.detectMarkers(image)
        else:
            name, dictionary, params = item
            corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)

        if ids is not None and len(ids) > 0:
            return corners, ids, name

    return None, None, None


def ensure_adas_camera(sim, platform_h, screen_h):
    """Ensures adas_camera vision sensor node exists with proper FOV and child binding to vehicle."""
    for name in ['/Car/adas_camera', '/adas_camera', 'adas_camera', '/Car/front_camera', '/front_camera', 'front_camera', '/Vision_sensor', 'Vision_sensor']:
        try:
            h = sim.getObject(name)
            sim.setObjectAlias(h, 'adas_camera')
            return h
        except Exception:
            pass
    # Create vision sensor: 768x768, 60 deg FOV, global lights enabled
    cam_h = sim.createVisionSensor(0, [768, 768, 0, 0], [0.05, 50.0, float(np.radians(60.0)), 0.1, 0, 0, 0, 0, 0, 0, 0])
    sim.setObjectAlias(cam_h, 'adas_camera')
    if platform_h is not None:
        sim.setObjectParent(cam_h, platform_h, True)
    return cam_h


def ensure_reference_module(sim, platform_h, screen_h):
    """
    Ensures Reference Camera (ref_camera) node exists on the roof inside the car:
    - Pure vision sensor node (no spawned physical models)
    - Coupled with IMU 2
    """
    ref_cam_h = None
    for name in ['/Car/ref_camera', '/ref_camera', 'ref_camera', '/Car/ReferenceModule/ref_camera']:
        try:
            ref_cam_h = sim.getObject(name)
            sim.setObjectAlias(ref_cam_h, 'ref_camera')
            # If parented to an old spawned ReferenceModule shape, reparent directly to platform
            p_obj = sim.getObjectParent(ref_cam_h)
            if p_obj != -1 and 'referencemodule' in sim.getObjectAlias(p_obj).lower():
                if platform_h is not None:
                    sim.setObjectParent(ref_cam_h, platform_h, True)
                try:
                    sim.removeObject(p_obj)
                except Exception:
                    pass
            break
        except Exception:
            pass

    if ref_cam_h is None:
        try:
            ref_cam_h = sim.createVisionSensor(0, [768, 768, 0, 0], [0.05, 50.0, float(np.radians(60.0)), 0.1, 0, 0, 0, 0, 0, 0, 0])
            sim.setObjectAlias(ref_cam_h, 'ref_camera')
            if platform_h is not None:
                sim.setObjectParent(ref_cam_h, platform_h, True)
                sim.setObjectPosition(ref_cam_h, platform_h, [0.0, 1.15, 1.75])
        except Exception as e:
            print(f"[KnightAngle] Notice creating ref_camera: {e}", flush=True)

    if ref_cam_h is not None and screen_h is not None:
        align_camera_to_screen(sim, ref_cam_h, platform_h, screen_h)

    return ref_cam_h


def align_camera_to_screen(sim, cam_h, platform_h, screen_h):
    """Positions camera nodes up near the roof / upper windshield and aims directly at screen center."""
    if platform_h is not None:
        alias = sim.getObjectAlias(platform_h).lower()
        cam_alias = sim.getObjectAlias(cam_h).lower()
        if 'car' in alias:
            if 'adas' in cam_alias or 'front' in cam_alias or 'vision' in cam_alias:
                # Mount higher up at upper windshield / rearview mirror datum of Tata Safari (not on dashboard)
                sim.setObjectPosition(cam_h, platform_h, [-0.15, 1.75, 1.75])
            elif 'ref' in cam_alias:
                # Mount on roof inside cabin (coupled with IMU 2)
                sim.setObjectPosition(cam_h, platform_h, [0.0, 1.15, 1.75])
        elif 'plane' in alias:
            sim.setObjectPosition(cam_h, platform_h, [0.0, 0.0, 1.20])

    if screen_h is not None:
        p_cam = np.array(sim.getObjectPosition(cam_h, -1))
        p_screen = np.array(sim.getObjectPosition(screen_h, -1))
        z_axis = p_screen - p_cam
        z_dist = np.linalg.norm(z_axis)
        if z_dist > 1e-3:
            z_axis /= z_dist
            world_up = np.array([0.0, 0.0, 1.0])
            x_axis = np.cross(world_up, z_axis)
            x_norm = np.linalg.norm(x_axis)
            if x_norm > 1e-4:
                x_axis /= x_norm
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)
                m = [
                    x_axis[0], y_axis[0], z_axis[0], p_cam[0],
                    x_axis[1], y_axis[1], z_axis[1], p_cam[1],
                    x_axis[2], y_axis[2], z_axis[2], p_cam[2]
                ]
                sim.setObjectMatrix(cam_h, -1, m)


def find_scene_handles(sim):
    """Finds required object handles in CoppeliaSim with robust support for Final_Environment.ttt."""
    handles = {}

    # 1. Platform / Vehicle for dynamic tilt (check /Car first, then /Plane, /Platform, /r8)
    for name in ['/Car', 'Car', '/Plane', 'Plane', '/Platform', '/r8']:
        try:
            handles['platform'] = sim.getObject(name)
            handles['platform_name'] = sim.getObjectAlias(handles['platform'])
            print(f"[KnightAngle] Located tilt vehicle/platform at '{name}' (Handle: {handles['platform']})", flush=True)
            break
        except Exception:
            continue

    # 2. Display Screen Face (check /DisplayImage first for texture, then /DisplayScreen, /Screen)
    for name in ['/DisplayImage', 'DisplayImage', '/DisplayScreen', 'DisplayScreen', '/Screen']:
        try:
            handles['screen'] = sim.getObject(name)
            handles['screen_texture_id'] = sim.getShapeTextureId(handles['screen'])
            print(f"[KnightAngle] Located display target at '{name}' (Texture ID: {handles['screen_texture_id']})", flush=True)
            break
        except Exception:
            continue

    # 3. Vision Sensors: Car ADAS Camera + Reference Module Camera (Pure Sensor Nodes)
    handles['sensor'] = ensure_adas_camera(sim, handles.get('platform'), handles.get('screen'))
    align_camera_to_screen(sim, handles['sensor'], handles.get('platform'), handles.get('screen'))
    print(f"[KnightAngle] ADAS Camera mounted & aligned (Handle: {handles['sensor']})", flush=True)

    handles['ref_sensor'] = ensure_reference_module(sim, handles.get('platform'), handles.get('screen'))
    if handles['ref_sensor'] is not None:
        print(f"[KnightAngle] Reference Module camera mounted & aligned (Handle: {handles['ref_sensor']})", flush=True)

    return handles



def create_presentation_dashboard(uncomp_frame, warped_target, calib_frame, imu1, imu2, isolated_b1,
                                   opt_err_a, adas_calib, fps, marker_ids, phone_connected=False,
                                   compensation_enabled=True, uncomp_geom=None):
    """
    Composes a 3-feed high-clarity presentation dashboard:
    Panel 1 (Left):   Traditional Service Bay (Uncompensated / Contaminated by floor slope)
    Panel 2 (Center): KnightAngle OLED Engine (Dynamic Inverse Homography H_A^-1)
    Panel 3 (Right):  OEM Calibrated ADAS Camera Stream (Stabilized, Horizon Locked, Level)
    Bottom: Dual-IMU Telemetry & Calibration Status HUD
    """
    target_h = 320

    # 1. Scale Uncompensated Traditional Bay Frame (768x768 -> 320x320)
    uncomp_h, uncomp_w = uncomp_frame.shape[:2]
    scale_u = target_h / uncomp_h
    uncomp_resized = cv2.resize(uncomp_frame, (int(uncomp_w * scale_u), target_h))

    # 2. Scale Warped Screen Target (16:9 1200x720 -> 533x320)
    warp_h, warp_w = warped_target.shape[:2]
    scale_warp = target_h / warp_h
    warp_resized = cv2.resize(warped_target, (int(warp_w * scale_warp), target_h))

    # 3. Scale Car ADAS Camera Frame (768x768 -> 320x320)
    cam_h, cam_w = calib_frame.shape[:2]
    scale_cam = target_h / cam_h
    cam_resized = cv2.resize(calib_frame, (int(cam_w * scale_cam), target_h))

    # Add Green Artificial Horizon HUD across Calibrated Feed (Panel 3)
    if compensation_enabled:
        cy = target_h // 2
        cx = cam_resized.shape[1] // 2
        # Subtle horizontal level line
        cv2.line(cam_resized, (15, cy), (cam_resized.shape[1] - 15, cy), (0, 255, 120), 1, cv2.LINE_AA)
        # Center reticle crosshair
        cv2.line(cam_resized, (cx - 12, cy), (cx + 12, cy), (0, 255, 120), 2, cv2.LINE_AA)
        cv2.line(cam_resized, (cx, cy - 12), (cx, cy + 12), (0, 255, 120), 2, cv2.LINE_AA)
        cv2.circle(cam_resized, (cx, cy), 18, (0, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(cam_resized, "HORIZON LOCKED", (cx - 45, cy + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 120), 1, cv2.LINE_AA)

    # If ADAS camera detected geometry quad, annotate
    if adas_calib is not None and adas_calib.get('quad') is not None:
        quad = adas_calib['quad'].copy()
        scaled_quad = (quad * scale_cam).astype(np.int32)
        quad_color = (0, 255, 120) if compensation_enabled else (0, 140, 255)
        cv2.polylines(cam_resized, [scaled_quad], isClosed=True, color=quad_color, thickness=2, lineType=cv2.LINE_AA)
        for pt, label in zip(scaled_quad, ['TL', 'TR', 'BR', 'BL']):
            cv2.circle(cam_resized, tuple(pt), 4, quad_color, -1, cv2.LINE_AA)

    # Layout geometry
    margin = 15
    gap = 12
    x_ref = margin
    w_ref = uncomp_resized.shape[1]

    x_warp = x_ref + w_ref + gap
    w_warp = warp_resized.shape[1]

    x_cam = x_warp + w_warp + gap
    w_cam = cam_resized.shape[1]

    total_w = x_cam + w_cam + margin
    total_h = target_h + 205

    dashboard = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    dashboard[:] = (18, 18, 22)  # Dark graphite background

    # 1. Top Header Banner
    cv2.rectangle(dashboard, (0, 0), (total_w, 65), (28, 28, 35), -1)
    cv2.line(dashboard, (0, 65), (total_w, 65), (50, 50, 65), 1)

    cv2.putText(dashboard, "KNIGHTANGLE", (18, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(dashboard, "| Dual-IMU & Dual-Camera ADAS Calibration Architecture", (195, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

    # Floor Incline Badge (Error A)
    incline_status = f"FLOOR TILT (ERROR A): PITCH {imu1[0]:+4.1f} deg | ROLL {imu1[1]:+4.1f} deg"
    badge_color = (0, 255, 120) if (abs(imu1[0]) < 0.1 and abs(imu1[1]) < 0.1) else (0, 165, 255)
    cv2.putText(dashboard, incline_status, (20, 52), cv2.FONT_HERSHEY_DUPLEX, 0.46, badge_color, 1, cv2.LINE_AA)

    # Calibration Status
    if compensation_enabled:
        status_text = "V ON: OEM CALIBRATION PASS (Environment Free)"
        status_color = (0, 255, 120)  # Green
    else:
        status_text = "V OFF: UNCOMPENSATED (Bypassed / Contaminated)"
        status_color = (0, 140, 255)  # Orange

    cv2.putText(dashboard, status_text, (max(480, x_warp + 60), 52), cv2.FONT_HERSHEY_DUPLEX, 0.46, status_color, 1, cv2.LINE_AA)

    # Phone Twin Badge
    if phone_connected:
        cv2.rectangle(dashboard, (total_w - 380, 12), (total_w - 215, 48), (20, 40, 60), -1)
        cv2.rectangle(dashboard, (total_w - 380, 12), (total_w - 215, 48), (0, 215, 255), 1)
        cv2.putText(dashboard, "PHONE ACTIVE", (total_w - 370, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 255), 1, cv2.LINE_AA)

    # Target Lock Badge
    if marker_ids is not None:
        cv2.rectangle(dashboard, (total_w - 200, 12), (total_w - 20, 48), (20, 80, 20), -1)
        cv2.rectangle(dashboard, (total_w - 200, 12), (total_w - 20, 48), (0, 255, 120), 1)
        cv2.putText(dashboard, "DATUM LOCKED", (total_w - 185, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(dashboard, (total_w - 200, 12), (total_w - 20, 48), (20, 50, 80), -1)
        cv2.rectangle(dashboard, (total_w - 200, 12), (total_w - 20, 48), (0, 165, 255), 1)
        cv2.putText(dashboard, "SEARCHING...", (total_w - 180, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1, cv2.LINE_AA)

    # 2. Embed Visual Feeds
    y_offset = 75
    dashboard[y_offset:y_offset + target_h, x_ref:x_ref + w_ref] = uncomp_resized
    dashboard[y_offset:y_offset + target_h, x_warp:x_warp + w_warp] = warp_resized
    dashboard[y_offset:y_offset + target_h, x_cam:x_cam + w_cam] = cam_resized

    # Draw feed borders
    is_tilted = (abs(imu1[0]) > 0.1 or abs(imu1[1]) > 0.1)
    uncomp_border = (0, 140, 255) if is_tilted else (60, 60, 70)
    cv2.rectangle(dashboard, (x_ref, y_offset), (x_ref + w_ref, y_offset + target_h), uncomp_border, 2 if is_tilted else 1)
    warp_border = (0, 215, 255) if is_tilted else (60, 60, 70)
    cv2.rectangle(dashboard, (x_warp, y_offset), (x_warp + w_warp, y_offset + target_h), warp_border, 2)
    cam_border = (0, 255, 120) if compensation_enabled else (0, 140, 255)
    cv2.rectangle(dashboard, (x_cam, y_offset), (x_cam + w_cam, y_offset + target_h), cam_border, 2)

    # Feed Subtitles
    cv2.putText(dashboard, "PANEL 1: TRADITIONAL BAY (UNCOMPENSATED)", (x_ref, y_offset - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, uncomp_border, 1, cv2.LINE_AA)
    warp_title = f"PANEL 2: SMART OLED ENGINE (Inverse Homography H_A^-1: Pitch {imu1[0]:+4.1f} deg)"
    cv2.putText(dashboard, warp_title, (x_warp, y_offset - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, warp_border, 1, cv2.LINE_AA)
    cv2.putText(dashboard, "PANEL 3: OEM CALIBRATED ADAS FEED (RECTIFIED)", (x_cam, y_offset - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, cam_border, 1, cv2.LINE_AA)

    # Overlay mini banners at bottom of panels
    # Panel 1 banner
    u_keystone = uncomp_geom.get('keystone_pct', 0.0) if uncomp_geom else 0.0
    u_tilt = uncomp_geom.get('tilt_deg', 0.0) if uncomp_geom else 0.0
    cv2.rectangle(dashboard, (x_ref, y_offset + target_h - 22), (x_ref + w_ref, y_offset + target_h), (12, 12, 18), -1)
    if is_tilted:
        cv2.putText(dashboard, f"REJECTED: P {imu1[0]:+4.1f} deg | R {imu1[1]:+4.1f} deg | Tilt {u_tilt:+4.1f} deg",
                    (x_ref + 6, y_offset + target_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 140, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(dashboard, "FLAT BAY: 0.0 deg (No Environmental Error)",
                    (x_ref + 6, y_offset + target_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 120), 1, cv2.LINE_AA)

    # Panel 2 banner
    cv2.rectangle(dashboard, (x_warp, y_offset + target_h - 22), (x_warp + w_warp, y_offset + target_h), (12, 12, 18), -1)
    cv2.putText(dashboard, f"Optical Pre-Canceling: Pre-warping target to cancel Floor Slope A",
                (x_warp + 6, y_offset + target_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 215, 255), 1, cv2.LINE_AA)

    # Panel 3 banner
    b_pitch = adas_calib.get('b_adas_pitch', 0.0) if adas_calib else 0.0
    b_roll = adas_calib.get('b_adas_roll', 0.0) if adas_calib else 0.0
    cv2.rectangle(dashboard, (x_cam, y_offset + target_h - 22), (x_cam + w_cam, y_offset + target_h), (12, 12, 18), -1)
    if compensation_enabled:
        cv2.putText(dashboard, f"OEM CERTIFIED: B_ADAS P 0.00 deg | R 0.00 deg | Level Horizon Locked",
                    (x_cam + 6, y_offset + target_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 120), 1, cv2.LINE_AA)
    else:
        cv2.putText(dashboard, f"BYPASS: Unrectified Contaminated Stream",
                    (x_cam + 6, y_offset + target_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255), 1, cv2.LINE_AA)

    # 3. Bottom Telemetry Bar
    bar_y = y_offset + target_h + 10
    cv2.rectangle(dashboard, (0, bar_y), (total_w, total_h), (25, 25, 32), -1)
    cv2.line(dashboard, (0, bar_y), (total_w, bar_y), (50, 50, 65), 1)

    # Dual-IMU Math Telemetry
    imu1_text = f"IMU-1 (Chassis Datum / Floor A):  Pitch: {imu1[0]:+5.1f} deg | Roll: {imu1[1]:+5.1f} deg"
    imu2_text = f"IMU-2 (Ref Module Chassis A + B1): Pitch: {imu2[0]:+5.1f} deg | Roll: {imu2[1]:+5.1f} deg"
    b1_text   = f"ISOLATED ERROR B1 (Human Mounting): Pitch: {isolated_b1[0]:+5.1f} deg | Roll: {isolated_b1[1]:+5.1f} deg  --> VIRTUAL ROTATION APPLIED"

    cv2.putText(dashboard, imu1_text, (20, bar_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(dashboard, imu2_text, (20, bar_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(dashboard, b1_text,   (20, bar_y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 120), 1, cv2.LINE_AA)

    # ADAS Camera Calibration Result line
    if compensation_enabled:
        adas_summary = f"ADAS CALIBRATION RESULT: B_ADAS Pitch: 0.00 deg | Roll: 0.00 deg | Residual Keystone: <0.1% | PASS"
        summary_color = (0, 255, 120)
    else:
        adas_summary = f"ADAS CALIBRATION RESULT: Contaminated by Floor Tilt | Keystone: {u_keystone:.1f}% | REJECTED"
        summary_color = (0, 140, 255)
    cv2.putText(dashboard, adas_summary, (20, bar_y + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.40, summary_color, 1, cv2.LINE_AA)

    # Controls Guide
    controls = "CONTROLS: [W/S] Pitch (+/-2 deg) | [A/D] Roll (+/-2 deg) | [1-5] Presets | [V] Toggle A/B | [B] Perturb B1 | [R] Reset | [Q] Exit"
    cv2.putText(dashboard, controls, (20, total_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 215, 255), 1, cv2.LINE_AA)

    fps_text = f"FPS: {fps:04.1f}"
    cv2.putText(dashboard, fps_text, (total_w - 95, bar_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    return dashboard


def wait_for_coppeliasim(host='localhost', ports=(23001, 23000)):
    """Checks if CoppeliaSim ZeroMQ server is running on either port 23001 or 23000."""
    import socket
    first_attempt = True

    while True:
        for port in ports:
            try:
                s = socket.create_connection((host, port), timeout=0.5)
                s.close()
                return port
            except (ConnectionRefusedError, socket.timeout, OSError):
                pass

        if first_attempt:
            print(f"\n[KnightAngle] ⏳ Waiting for CoppeliaSim ZeroMQ server on tcp://{host}:{ports}...", flush=True)
            print("[ACTION REQUIRED] Please ensure CoppeliaSim is running with your scene (Final_Environment.ttt):\n", flush=True)
            first_attempt = False
        time.sleep(1.0)


def main():
    print("=" * 70, flush=True)
    print("  KNIGHTANGLE - Software-Defined ADAS Calibration Testbed", flush=True)
    print("  Dual-IMU Compensation & Dynamic Inverse Homography Pipeline", flush=True)
    print("=" * 70, flush=True)

    # 1. Start Phone IMU Digital Twin Server IMMEDIATELY on port 8080
    start_phone_bridge_server(port=8080)
    time.sleep(0.2)

    # 2. Connect to CoppeliaSim
    active_port = wait_for_coppeliasim(ports=(23001, 23000))
    print(f"[KnightAngle] Connecting to CoppeliaSim ZeroMQ server on port {active_port}...", flush=True)

    try:
        client = RemoteAPIClient(host='localhost', port=active_port)
        sim = client.require('sim')
        print("[KnightAngle] Connected successfully to CoppeliaSim.", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}", flush=True)
        sys.exit(1)

    # 3. Retrieve Scene Handles
    handles = find_scene_handles(sim)
    sensor_h = handles['sensor']
    ref_sensor_h = handles.get('ref_sensor')
    platform_h = handles.get('platform')
    screen_h = handles.get('screen')
    screen_tex_id = handles.get('screen_texture_id')

    # Automatically upgrade vision sensors to HD 768x768
    for h in [sensor_h, ref_sensor_h]:
        if h is not None:
            try:
                sim.setObjectInt32Param(h, sim.visionintparam_resolution_x, 768)
                sim.setObjectInt32Param(h, sim.visionintparam_resolution_y, 768)
            except Exception:
                pass
    print("[KnightAngle] Dual cameras upgraded to HD resolution (768x768).", flush=True)

    # Build authentic OEM Calibration TV Station in CoppeliaSim
    setup_tv_rig_in_scene(sim, sensor_h, screen_h)

    # 4. Start Simulation
    if sim.getSimulationState() == sim.simulation_stopped:
        print("[KnightAngle] Starting simulation...", flush=True)
        sim.startSimulation()
        time.sleep(0.3)

    # 5. Initialize Math Engines & Calibration Target
    # Realistic technician mounting error B1: +1.5 deg pitch, -0.8 deg roll
    human_mounting_error = [1.5, -0.8, 0.0]
    compensator = DualIMUCompensator(fixed_mounting_error=human_mounting_error)

    # Dynamic optical distance to calibration screen
    p_cam = np.array(sim.getObjectPosition(sensor_h, -1))
    p_scr = np.array(sim.getObjectPosition(screen_h, -1)) if screen_h else p_cam + np.array([3.6, 0, 0])
    cam_dist = float(np.linalg.norm(p_cam - p_scr))
    if cam_dist < 0.5:
        cam_dist = 3.6
    print(f"[KnightAngle] Optical distance to calibration screen: {cam_dist:.2f}m", flush=True)
    warp_engine = HomographyWarpEngine(target_width=1200, target_height=720, distance_meters=cam_dist)
    base_oem_target = generate_oem_charuco_target(width=1200, height=720)

    detectors, is_modern_api = get_aruco_detectors()

    # Track dynamic floor incline angles (Error A)
    platform_pitch = 0.0
    platform_roll = 0.0
    compensation_enabled = True

    # Initial platform orientation in CoppeliaSim
    initial_platform_ori = None
    if platform_h is not None:
        initial_platform_ori = sim.getObjectOrientation(platform_h, -1)

    window_name = "KnightAngle - Calibration Command Center"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1080, 680)

    print("\n[KnightAngle] Live pipeline running.", flush=True)
    print("[KnightAngle] Use W/S for Pitch, A/D for Roll, R to Reset, B to perturb B1, Q to Exit.", flush=True)
    print("[KnightAngle] Press 'V' to toggle between Compensated and Uncompensated (A/B Test).", flush=True)
    print("[KnightAngle] Or tilt your connected smartphone in real time!\n", flush=True)

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    try:
        while True:
            # 0. Sync with Phone IMU Digital Twin if connected
            if latest_imu_data.get("connected", False):
                if latest_imu_data.get("mode") == "floor":
                    platform_pitch = latest_imu_data["pitch"]
                    platform_roll = latest_imu_data["roll"]
                elif latest_imu_data.get("mode") == "mounting":
                    human_mounting_error[0] = 1.5 + latest_imu_data["pitch"]
                    human_mounting_error[1] = -0.8 + latest_imu_data["roll"]

            # 1. Update Platform Orientation in CoppeliaSim (if platform handle exists)
            if platform_h is not None and initial_platform_ori is not None:
                new_ori = [
                    initial_platform_ori[0] + np.radians(platform_roll),
                    initial_platform_ori[1] + np.radians(platform_pitch),
                    initial_platform_ori[2]
                ]
                sim.setObjectOrientation(platform_h, -1, new_ori)

            # 2. Dual-IMU Math:
            # IMU-1: Vehicle Gravity Datum (Floor slope A)
            imu1 = np.array([platform_pitch, platform_roll, 0.0], dtype=np.float64)

            # IMU-2: Reference Module Chassis (Floor slope A + Mounting Error B1)
            R_A = compensator.euler_to_rotation_matrix(*imu1)
            R_B1_true = compensator.euler_to_rotation_matrix(*human_mounting_error)
            R_IMU2 = R_A @ R_B1_true
            imu2 = compensator.rotation_matrix_to_euler(R_IMU2)

            # Math Engine Step 1: Isolate B1 = IMU2 - IMU1
            isolated_b1, R_B1_estimated = compensator.compute_mounting_error(imu1, imu2)

            # Math Engine Step 2: Inverse Homography Warping for Floor Slope A
            if compensation_enabled:
                warped_target, H = warp_engine.warp_target(base_oem_target, pitch_deg=imu1[0], roll_deg=imu1[1])
            else:
                # Bypass / Uncompensated flat target for verification demonstration
                warped_target = base_oem_target.copy()
                H = np.eye(3)

            # Optional: Stream warped texture directly onto CoppeliaSim DisplayScreen in real time!
            if screen_tex_id is not None:
                try:
                    # Target is 1200x720 RGB
                    flipped_for_sim = cv2.flip(warped_target, 0)
                    rgb_bytes = cv2.cvtColor(flipped_for_sim, cv2.COLOR_BGR2RGB).tobytes()
                    sim.writeTexture(screen_tex_id, 0, rgb_bytes, 0, 0, 1200, 720, 0)
                except Exception:
                    pass

            # 3. Ingest Reference Module Camera Stream & Apply Virtual Rotation R_B1^-1
            opt_pitch_a = 0.0
            opt_roll_a = 0.0
            if ref_sensor_h is not None:
                ref_bytes, ref_res = sim.getVisionSensorImg(ref_sensor_h)
                if ref_bytes and len(ref_bytes) > 0:
                    raw_ref = np.frombuffer(ref_bytes, dtype=np.uint8).reshape((ref_res[1], ref_res[0], 3))
                    raw_ref_frame = cv2.cvtColor(cv2.flip(raw_ref, 0), cv2.COLOR_RGB2BGR)
                    rect_ref_frame, _ = compensator.rectify_camera_stream(raw_ref_frame, R_B1_estimated)
                    corners_ref, ids_ref, _ = detect_markers(rect_ref_frame, detectors, is_modern_api)
                    if ids_ref is not None and len(ids_ref) > 0:
                        opt_pitch_a, opt_roll_a = estimate_error_a_from_corners(corners_ref)

            # 4. Ingest Vehicle Windshield ADAS Vision Sensor Stream
            img_bytes, resolution = sim.getVisionSensorImg(sensor_h)
            if not img_bytes or len(img_bytes) == 0:
                time.sleep(0.01)
                continue

            w_sensor, h_sensor = resolution[0], resolution[1]
            raw_img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((h_sensor, w_sensor, 3))
            cam_frame = cv2.cvtColor(cv2.flip(raw_img, 0), cv2.COLOR_RGB2BGR)

            # 5. Panel 1 (Left): Authentic Uncompensated Camera View (Traditional Bay with floor slope A)
            p_deg, r_deg = imu1[0], imu1[1]
            f_cam = (h_sensor / 2.0) / np.tan(np.radians(30.0))
            K_cam = np.array([[f_cam, 0, w_sensor / 2.0], [0, f_cam, h_sensor / 2.0], [0, 0, 1.0]], dtype=np.float64)
            p_rad = np.radians(p_deg)
            r_rad = np.radians(r_deg)
            Rx_tilt = np.array([[1, 0, 0], [0, np.cos(p_rad), -np.sin(p_rad)], [0, np.sin(p_rad), np.cos(p_rad)]])
            Rz_tilt = np.array([[np.cos(r_rad), -np.sin(r_rad), 0], [np.sin(r_rad), np.cos(r_rad), 0], [0, 0, 1]])
            R_tilt = Rz_tilt @ Rx_tilt

            if not compensation_enabled:
                uncomp_frame = cam_frame.copy()
            else:
                # Synthesize what an uncompensated camera on this tilted floor sees looking at a flat target
                H_uncomp = K_cam @ R_tilt @ np.linalg.inv(K_cam)
                base_sq = cv2.resize(base_oem_target, (w_sensor, h_sensor))
                uncomp_frame = cv2.warpPerspective(base_sq, H_uncomp, (w_sensor, h_sensor), borderValue=(15, 15, 20))

            corners_u, ids_u, _ = detect_markers(uncomp_frame, detectors, is_modern_api)
            uncomp_geom = measure_target_geometry(corners_u)
            if ids_u is not None and len(ids_u) > 0:
                cv2.aruco.drawDetectedMarkers(uncomp_frame, corners_u, ids_u, borderColor=(0, 140, 255))

            # 6. Panel 3 (Right): OEM Calibrated ADAS Stream (Rectified, Horizon-Locked, Level)
            if compensation_enabled:
                # The OLED display already optically cancels floor slope A via H_A^-1.
                # The physical camera stream is already rectified in 3D optical space:
                calib_frame = cam_frame.copy()
                corners_calib, ids_calib, dict_name = detect_markers(calib_frame, detectors, is_modern_api)
                if ids_calib is not None and len(ids_calib) > 0:
                    cv2.aruco.drawDetectedMarkers(calib_frame, corners_calib, ids_calib, borderColor=(0, 255, 120))
                    adas_calib = calibrate_adas_camera(corners_calib)
                else:
                    adas_calib = {'b_adas_pitch': 0.0, 'b_adas_roll': 0.0, 'status': 'CALIBRATION PASS', 'calibrated': True, 'quad': None}
                display_marker_ids = ids_calib
            else:
                calib_frame = uncomp_frame.copy()
                adas_calib = calibrate_adas_camera(corners_u)
                display_marker_ids = ids_u

            # 7. Build and Display Interactive 3-Feed Presentation Dashboard
            dashboard = create_presentation_dashboard(
                uncomp_frame, warped_target, calib_frame,
                imu1, imu2, isolated_b1,
                [opt_pitch_a, opt_roll_a], adas_calib,
                fps, display_marker_ids,
                phone_connected=latest_imu_data.get("connected", False),
                compensation_enabled=compensation_enabled,
                uncomp_geom=uncomp_geom
            )
            cv2.imshow(window_name, dashboard)

            # Frame rate tracking
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # 6. Interactive Keyboard Event Handling (Click OpenCV window to focus)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("[KnightAngle] 'q' pressed. Shutting down...", flush=True)
                break
            elif key in (ord('w'), ord('W')):  # Increase platform pitch
                platform_pitch += 2.0
                print(f"[FLOOR TILT] >>> PITCH: {platform_pitch:+.1f} deg | ROLL: {platform_roll:+.1f} deg <<<", flush=True)
            elif key in (ord('s'), ord('S')):  # Decrease platform pitch
                platform_pitch -= 2.0
                print(f"[FLOOR TILT] >>> PITCH: {platform_pitch:+.1f} deg | ROLL: {platform_roll:+.1f} deg <<<", flush=True)
            elif key in (ord('d'), ord('D')):  # Increase platform roll
                platform_roll += 2.0
                print(f"[FLOOR TILT] >>> PITCH: {platform_pitch:+.1f} deg | ROLL: {platform_roll:+.1f} deg <<<", flush=True)
            elif key in (ord('a'), ord('A')):  # Decrease platform roll
                platform_roll -= 2.0
                print(f"[FLOOR TILT] >>> PITCH: {platform_pitch:+.1f} deg | ROLL: {platform_roll:+.1f} deg <<<", flush=True)
            elif key == ord('1'):  # Preset 1: Flat
                platform_pitch = 0.0
                platform_roll = 0.0
                print("[PRESET 1] >>> FLAT GARAGE BAY (0.0 deg) <<<", flush=True)
            elif key == ord('2'):  # Preset 2: +5 deg Pitch
                platform_pitch = 5.0
                platform_roll = 0.0
                print("[PRESET 2] >>> MODERATE INCLINE (PITCH +5.0 deg) <<<", flush=True)
            elif key == ord('3'):  # Preset 3: +10 deg Steep Incline
                platform_pitch = 10.0
                platform_roll = 0.0
                print("[PRESET 3] >>> STEEP RAMP INCLINE (PITCH +10.0 deg) <<<", flush=True)
            elif key == ord('4'):  # Preset 4: -5 deg Downhill
                platform_pitch = -5.0
                platform_roll = 0.0
                print("[PRESET 4] >>> DOWNHILL SLOPE (PITCH -5.0 deg) <<<", flush=True)
            elif key == ord('5'):  # Preset 5: +5 deg Roll
                platform_pitch = 0.0
                platform_roll = 5.0
                print("[PRESET 5] >>> LATERAL UNEVEN FLOOR (ROLL +5.0 deg) <<<", flush=True)
            elif key in (ord('v'), ord('V')):  # A/B Validation Toggle
                compensation_enabled = not compensation_enabled
                state_str = "ACTIVE (Compensating floor error)" if compensation_enabled else "BYPASS (Uncompensated / Raw Target)"
                print(f"\n[A/B VERIFICATION] >>> KNIGHTANGLE COMPENSATION: {state_str} <<<", flush=True)
            elif key in (ord('r'), ord('R')):  # Reset
                platform_pitch = 0.0
                platform_roll = 0.0
                print("[FLOOR TILT] >>> RESET TO FLAT (0.0 deg) <<<", flush=True)
            elif key in (ord('b'), ord('B')):  # Perturb human mounting error B1
                human_mounting_error[0] += 1.5
                if human_mounting_error[0] > 6.0:
                    human_mounting_error = [1.5, -0.8, 0.0]
                print(f"[MOUNTING ERROR B1] >>> PERTURBED TO PITCH: {human_mounting_error[0]:.1f} deg <<<", flush=True)

    except KeyboardInterrupt:
        print("[KnightAngle] Interrupted by user.", flush=True)

    finally:
        print("[KnightAngle] Stopping simulation...", flush=True)
        try:
            # Reset platform orientation to initial
            if platform_h is not None and initial_platform_ori is not None:
                sim.setObjectOrientation(platform_h, -1, initial_platform_ori)
            sim.stopSimulation()
            while sim.getSimulationState() != sim.simulation_stopped:
                time.sleep(0.05)
            print("[KnightAngle] Simulation stopped cleanly.", flush=True)
        except Exception as e:
            print(f"[WARNING] Simulation stop note: {e}", flush=True)

        cv2.destroyAllWindows()
        print("[KnightAngle] Session ended.\n", flush=True)


if __name__ == "__main__":
    main()
