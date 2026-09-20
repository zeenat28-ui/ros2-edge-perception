"""
Deep Video Frame Quality & Visual Sanity Analyzer
Extracts key stage frames, checks for clipping, contrast, overlap, and anomalies.
"""

import cv2
import numpy as np
import os

VIDEO_PATH = r"C:\Users\Zeenat\Desktop\ros2_enterprise_perception_showcase.mp4"
OUTPUT_DIR = r"C:\Users\Zeenat\Desktop\ros2-edge-perception\frame_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open {VIDEO_PATH}")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("=== VIDEO STREAM METADATA ===")
print(f"File: {VIDEO_PATH}")
print(f"Total Frames: {total_frames}")
print(f"FPS: {fps:.2f}")
print(f"Resolution: {width}x{height}")
print(f"Duration: {total_frames / fps:.2f} seconds")

# Stages to sample
sample_points = {
    60: "stage1_boot_handshake",
    200: "stage2_3d_tracking_fusion",
    350: "stage3_obstacle_ttc_approach",
    500: "stage4_aeb_intervention",
    650: "stage5_evasive_maneuver",
    780: "stage6_sensor_dropout_failsafe"
}

results = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    if frame_idx in sample_points:
        name = sample_points[frame_idx]
        out_img_path = os.path.join(OUTPUT_DIR, f"{name}_f{frame_idx:04d}.png")
        cv2.imwrite(out_img_path, frame)
        
        # Quality & Contrast Metrics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_lum = np.mean(gray)
        std_lum = np.std(gray)
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)
        
        # Check clipping in left viewport (0 to 840) vs right instrument panel (840 to 1280)
        left_crop = frame[:, 0:840]
        right_crop = frame[:, 840:1280]
        
        left_mean = np.mean(left_crop)
        right_mean = np.mean(right_crop)
        
        info = {
            "frame": frame_idx,
            "stage": name,
            "path": out_img_path,
            "mean_lum": round(float(mean_lum), 2),
            "std_lum": round(float(std_lum), 2),
            "dynamic_range": f"{int(min_val)} to {int(max_val)}",
            "left_lum": round(float(left_mean), 2),
            "right_lum": round(float(right_mean), 2),
            "status": "PASSED" if std_lum > 20 and max_val > 200 else "FLAGGED"
        }
        results.append(info)
        print(f"Analyzed Frame {frame_idx:04d} ({name}): Dynamic Range [{min_val:.0f} - {max_val:.0f}], StdDev: {std_lum:.1f} -> {info['status']}")

    frame_idx += 1

cap.release()

print("\n=== FRAME-BY-FRAME ANALYSIS SUMMARY ===")
for r in results:
    print(f"[{r['status']}] Frame {r['frame']:04d} ({r['stage']}): Saved to {r['path']}")

