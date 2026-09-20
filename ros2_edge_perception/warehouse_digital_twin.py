#!/usr/bin/env python3
"""
3D Warehouse Simulation Environment for Closed-Loop Autonomy Testing.

Simulates an industrial warehouse arena populated with static and dynamic obstacles:
- Storage racks, pallet staging areas, and cargo containers
- Dynamic obstacle agents (moving forklift trucks, workers)
- Low-lying hazards (3cm floor cables) and negative obstacles (loading dock cliff edges)
- Virtual 16-channel LiDAR and forward-facing RGB-D camera sensor streams
- 4-pane diagnostic visualization (world view, camera view, occupancy/costmap, telemetry)
"""

import os
import sys
import time
import math
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ros2_edge_perception.neural_vla_engine import NeuralVLAEngine, VLAActionChunk
from ros2_edge_perception.neural_occupancy_network import Neural3DOccupancyNetwork, VoxelSemanticClass
from ros2_edge_perception.cuda_mppi_optimizer import ParallelMPPIOptimizer
from ros2_edge_perception.near_ground_hazard_detector import NearGroundHazardDetector
from ros2_edge_perception.iso3691_safety_field import ISO3691DynamicSafetySupervisor, SafetyZone, SafetyInterlockState


class WarehouseDigitalTwin:
    """3D Industrial Logistics Warehouse Digital Twin & Video Generator."""

    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pane_w = width // 2
        self.pane_h = height // 2

        # Initialize Physical AI Engines
        self.vla_engine = NeuralVLAEngine()
        self.occupancy_net = Neural3DOccupancyNetwork()
        self.mppi_optimizer = ParallelMPPIOptimizer(num_samples=10000, horizon=20)
        self.hazard_detector = NearGroundHazardDetector()
        self.safety_supervisor = ISO3691DynamicSafetySupervisor()

        # Warehouse 3D World Dimensions (meters)
        self.world_len = 30.0   # X: 0 to 30m
        self.world_width = 16.0 # Y: -8 to +8m

        # Robot State in 3D World: [x, y, theta, v, w]
        self.robot_pos = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        self.robot_vel = np.array([0.0, 0.0], dtype=np.float32)  # [linear_v, angular_w]
        self.robot_target = np.array([22.0, 0.0, 0.0], dtype=np.float32)

        # Dynamic Forklift: starts at x=12.0, y=5.0, moves across to y=-5.0
        self.forklift_pos = np.array([12.5, 4.5, 0.0], dtype=np.float32)
        self.forklift_vel = np.array([0.0, -0.75, 0.0], dtype=np.float32)

        # Static Hazards
        self.cable_pos = np.array([17.0, 0.0, 0.03], dtype=np.float32)  # 3cm cable at x=17m
        self.dock_edge_x = 24.0  # Loading dock cliff edge at x=24m

    def run_mission_simulation(self, output_video_path: str, duration_sec: float = 12.0):
        """Executes full autonomous mission and records 4-pane broadcast video."""
        print(f"[AURA-Drive Digital Twin] Starting 3D Simulation for {duration_sec}s...")
        total_frames = int(duration_sec * self.fps)
        dt = 1.0 / self.fps

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.width, self.height))

        # Natural language task instruction
        task_prompt = "Navigate through aisle 2, bypass crossing forklift on left, inspect low cable, dock at station 4"
        print(f"  Task Prompt: \"{task_prompt}\"")

        # Initial dummy observation
        dummy_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy_rgb[80:140, 90:130] = [0, 165, 255]  # Simulated orange forklift

        for frame_idx in range(total_frames):
            sim_time = frame_idx * dt

            # 1. Update Dynamic Obstacles (Forklift)
            if 1.5 <= sim_time <= 8.0:
                self.forklift_pos += self.forklift_vel * dt

            # 2. Neural VLA Forward Pass (every 5 frames / 6 Hz)
            if frame_idx % 5 == 0:
                vla_chunk = self.vla_engine.predict_action_chunk(dummy_rgb, task_prompt, self.robot_pos)

            # 3. Virtual 3D LiDAR Raycasting & Point Cloud Generation
            lidar_pts = self._generate_virtual_lidar(self.robot_pos)

            # 4. Neural 3D Occupancy Network (OccNet) Update
            dynamic_agents = [{
                "centroid_3d": [self.forklift_pos[0] - self.robot_pos[0],
                                self.forklift_pos[1] - self.robot_pos[1], 1.0],
                "bbox_dimensions": [2.4, 1.2, 2.0]
            }]
            drop_offs = [{
                "centroid_3d": [self.dock_edge_x - self.robot_pos[0] + 1.0, 0.0, -0.3],
                "bbox_dimensions": [2.0, 4.0, 0.4]
            }] if (self.dock_edge_x - self.robot_pos[0]) < 8.0 else []

            occ_grid = self.occupancy_net.predict_occupancy(
                point_cloud=lidar_pts,
                dynamic_agents=dynamic_agents,
                negative_drop_offs=drop_offs
            )

            # 5. Near-Ground Hazard Detector (Cables & Drop-offs)
            hazards, ground_plane = self.hazard_detector.detect_hazards(lidar_pts)

            # 6. Parallel MPPI Optimizer (10,000 Rollouts)
            mppi_res = self.mppi_optimizer.optimize(
                current_state=np.array([0.0, 0.0, 0.0, self.robot_vel[0], self.robot_vel[1]]),
                target_waypoints=vla_chunk.waypoints,
                bev_costmap=occ_grid.bev_costmap,
                costmap_origin=(occ_grid.origin_xyz[0], occ_grid.origin_xyz[1]),
                costmap_resolution=occ_grid.voxel_size_m
            )

            # 7. ISO 3691-4 Dynamic Safety Field Enforcement
            obs_for_safety = [h.to_dict() for h in hazards]
            # Add dynamic forklift to safety supervisor
            obs_for_safety.append({
                "hazard_id": 888,
                "centroid_3d": [self.forklift_pos[0] - self.robot_pos[0],
                                self.forklift_pos[1] - self.robot_pos[1], 0.8]
            })

            cmd_v = mppi_res.optimal_controls[0, 0]
            cmd_w = mppi_res.optimal_controls[0, 1]

            safe_v, safe_w, zone, report = self.safety_supervisor.evaluate_safety_and_clamp(
                command_linear_v=cmd_v,
                command_angular_w=cmd_w,
                current_linear_v=self.robot_vel[0],
                obstacles=obs_for_safety
            )

            # 8. Robot Kinematic Step (World Frame)
            self.robot_vel[0] = safe_v
            self.robot_vel[1] = safe_w
            self.robot_pos[0] += safe_v * math.cos(self.robot_pos[2]) * dt
            self.robot_pos[1] += safe_v * math.sin(self.robot_pos[2]) * dt
            self.robot_pos[2] += safe_w * dt

            # 9. Render the 4 Cockpit Panes
            pane1 = self._render_isometric_view(sim_time)
            pane2 = self._render_fpv_view(sim_time, hazards)
            pane3 = self._render_occupancy_mppi_view(occ_grid, mppi_res)
            pane4 = self._render_vla_safety_cockpit(sim_time, task_prompt, vla_chunk, zone, mppi_res.latency_ms)

            # Composite into Full 1080p Canvas
            full_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            full_frame[0:self.pane_h, 0:self.pane_w] = pane1
            full_frame[0:self.pane_h, self.pane_w:self.width] = pane2
            full_frame[self.pane_h:self.height, 0:self.pane_w] = pane3
            full_frame[self.pane_h:self.height, self.pane_w:self.width] = pane4

            # Draw divider lines
            cv2.line(full_frame, (self.pane_w, 0), (self.pane_w, self.height), (60, 60, 60), 2)
            cv2.line(full_frame, (0, self.pane_h), (self.width, self.pane_h), (60, 60, 60), 2)

            out.write(full_frame)

            if frame_idx % 30 == 0:
                pct = int((frame_idx / total_frames) * 100)
                print(f"  Rendered frame {frame_idx}/{total_frames} ({pct}%) - Robot X={self.robot_pos[0]:.2f}m, Zone={zone.value}")

        out.release()
        print(f"[AURA-Drive Digital Twin] Video rendered successfully: {output_video_path}")

    def _generate_virtual_lidar(self, robot_pos: np.ndarray) -> np.ndarray:
        """Simulates a spinning 16-beam 3D LiDAR point cloud in robot frame."""
        pts = []
        rx, ry, rtheta = robot_pos[0], robot_pos[1], robot_pos[2]

        # 1. Concrete Floor Points
        num_floor = 1200
        fx = np.random.uniform(0.5, 9.0, num_floor)
        fy = np.random.uniform(-3.5, 3.5, num_floor)
        fz = np.random.normal(0.0, 0.005, num_floor)
        floor = np.column_stack([fx, fy, fz])
        pts.append(floor)

        # 2. Forklift Points (if within 10m)
        dx_f = self.forklift_pos[0] - rx
        dy_f = self.forklift_pos[1] - ry
        # Transform into robot frame
        fx_rob = dx_f * math.cos(-rtheta) - dy_f * math.sin(-rtheta)
        fy_rob = dx_f * math.sin(-rtheta) + dy_f * math.cos(-rtheta)

        if 0.5 <= fx_rob <= 9.0 and abs(fy_rob) <= 3.5:
            num_f_pts = 180
            px = np.random.uniform(fx_rob - 0.6, fx_rob + 0.6, num_f_pts)
            py = np.random.uniform(fy_rob - 0.5, fy_rob + 0.5, num_f_pts)
            pz = np.random.uniform(0.1, 1.8, num_f_pts)
            pts.append(np.column_stack([px, py, pz]))

        # 3. Near-Ground Cable Points (at x=17m in world)
        dx_c = self.cable_pos[0] - rx
        dy_c = self.cable_pos[1] - ry
        cx_rob = dx_c * math.cos(-rtheta) - dy_c * math.sin(-rtheta)
        cy_rob = dx_c * math.sin(-rtheta) + dy_c * math.cos(-rtheta)

        if 0.5 <= cx_rob <= 8.0:
            num_c_pts = 45
            px = np.random.uniform(cx_rob - 0.04, cx_rob + 0.04, num_c_pts)
            py = np.random.uniform(-1.0, 1.0, num_c_pts)
            pz = np.random.uniform(0.025, 0.045, num_c_pts)
            pts.append(np.column_stack([px, py, pz]))

        # 4. Dock Drop-Off Edge (at x=24m in world)
        dx_dock = self.dock_edge_x - rx
        if 1.0 <= dx_dock <= 7.0:
            num_pit_pts = 120
            px = np.random.uniform(dx_dock + 0.2, dx_dock + 3.0, num_pit_pts)
            py = np.random.uniform(-2.0, 2.0, num_pit_pts)
            pz = np.random.normal(-0.35, 0.02, num_pit_pts)
            pts.append(np.column_stack([px, py, pz]))

        return np.vstack(pts).astype(np.float32)

    def _render_isometric_view(self, sim_time: float) -> np.ndarray:
        """Pane 1: 3D Isometric View of the Warehouse Digital Twin."""
        canvas = np.zeros((self.pane_h, self.pane_w, 3), dtype=np.uint8)
        # Background dark industrial theme
        canvas[:] = (22, 25, 30)

        # Isometric projection matrix: (x_world, y_world, z_world) -> (u, v)
        # Center origin at (pane_w // 2 - 100, pane_h // 2 + 100)
        u0 = self.pane_w // 2 - 120
        v0 = self.pane_h // 2 + 120
        scale = 22.0  # pixels per meter

        def to_iso(x, y, z):
            # Isometric angles: 30 deg
            u = u0 + (x - y) * math.cos(math.radians(30)) * scale
            v = v0 + (x + y) * math.sin(math.radians(30)) * scale * 0.5 - z * scale
            return int(u), int(v)

        # 1. Draw Grid Floor
        for x in range(0, 26, 2):
            p1 = to_iso(x, -6, 0)
            p2 = to_iso(x, 6, 0)
            cv2.line(canvas, p1, p2, (38, 44, 52), 1)
        for y in range(-6, 7, 2):
            p1 = to_iso(0, y, 0)
            p2 = to_iso(25, y, 0)
            cv2.line(canvas, p1, p2, (38, 44, 52), 1)

        # 2. Draw Storage Racks along both sides (y = -5m and y = +5m)
        for rack_y in [-5.0, 5.0]:
            for rx in range(4, 22, 4):
                # Upright posts (height 3.5m)
                b1 = to_iso(rx, rack_y - 0.8, 0)
                t1 = to_iso(rx, rack_y - 0.8, 3.5)
                b2 = to_iso(rx + 3.0, rack_y - 0.8, 0)
                t2 = to_iso(rx + 3.0, rack_y - 0.8, 3.5)
                cv2.line(canvas, b1, t1, (80, 100, 120), 2)
                cv2.line(canvas, b2, t2, (80, 100, 120), 2)
                cv2.line(canvas, t1, t2, (0, 140, 255), 2)  # Orange rack beam
                # Draw pallet box
                box_c = to_iso(rx + 1.5, rack_y - 0.8, 1.0)
                cv2.rectangle(canvas, (box_c[0]-14, box_c[1]-10), (box_c[0]+14, box_c[1]+10), (45, 80, 110), -1)

        # 3. Draw Loading Dock Cliff Edge (at x = 24m)
        c1 = to_iso(self.dock_edge_x, -6, 0)
        c2 = to_iso(self.dock_edge_x, 6, 0)
        cv2.line(canvas, c1, c2, (0, 0, 255), 3)  # Bright Red Edge
        # Pit depth hatching
        cp1 = to_iso(self.dock_edge_x, -6, -1.0)
        cp2 = to_iso(self.dock_edge_x, 6, -1.0)
        cv2.line(canvas, cp1, cp2, (0, 0, 150), 2)

        # 4. Draw Loose Floor Cable (at x = 17m)
        cb1 = to_iso(self.cable_pos[0], -1.5, 0.03)
        cb2 = to_iso(self.cable_pos[0], 1.5, 0.03)
        cv2.line(canvas, cb1, cb2, (0, 220, 255), 3)  # Yellow Hazard Line

        # 5. Draw Dynamic Forklift
        fk_pt = to_iso(self.forklift_pos[0], self.forklift_pos[1], 0)
        cv2.circle(canvas, fk_pt, 12, (0, 140, 255), -1)
        cv2.putText(canvas, "FORKLIFT", (fk_pt[0] - 28, fk_pt[1] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 255), 1)

        # 6. Draw Robot AMR
        rb_pt = to_iso(self.robot_pos[0], self.robot_pos[1], 0)
        cv2.circle(canvas, rb_pt, 10, (0, 255, 120), -1)
        # Heading arrow
        h_x = self.robot_pos[0] + 1.2 * math.cos(self.robot_pos[2])
        h_y = self.robot_pos[1] + 1.2 * math.sin(self.robot_pos[2])
        h_pt = to_iso(h_x, h_y, 0)
        cv2.arrowedLine(canvas, rb_pt, h_pt, (0, 255, 255), 2, tipLength=0.3)

        # Overlay Header
        cv2.rectangle(canvas, (10, 10), (320, 42), (10, 15, 20), -1)
        cv2.putText(canvas, "3D DIGITAL TWIN: ISOMETRIC", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
        return canvas

    def _render_fpv_view(self, sim_time: float, hazards: List) -> np.ndarray:
        """Pane 2: Robot First-Person View (FPV) RGB-D Camera with 3D Bounding Boxes."""
        canvas = np.zeros((self.pane_h, self.pane_w, 3), dtype=np.uint8)
        # Perspective warehouse interior
        canvas[:] = (30, 35, 42)

        # Concrete floor gradient
        cv2.rectangle(canvas, (0, self.pane_h // 2), (self.pane_w, self.pane_h), (45, 48, 55), -1)
        # Ceiling & lights
        for lx in range(120, self.pane_w, 200):
            cv2.line(canvas, (lx, 20), (lx - 60, self.pane_h // 2 - 30), (160, 170, 180), 1)

        # Perspective Aisle Lines
        cv2.line(canvas, (self.pane_w // 2 - 30, self.pane_h // 2), (60, self.pane_h), (200, 180, 0), 2)
        cv2.line(canvas, (self.pane_w // 2 + 30, self.pane_h // 2), (self.pane_w - 60, self.pane_h), (200, 180, 0), 2)

        # Project Forklift in FPV if ahead
        rx, ry, rth = self.robot_pos
        dx = self.forklift_pos[0] - rx
        dy = self.forklift_pos[1] - ry
        # In robot frame: x forward, y left
        fx_rob = dx * math.cos(-rth) - dy * math.sin(-rth)
        fy_rob = dx * math.sin(-rth) + dy * math.cos(-rth)

        if fx_rob > 0.8:
            # Perspective projection: u = cx - (y / x) * f, v = cy + (z / x) * f
            f_scale = 350.0 / fx_rob
            u = int(self.pane_w / 2 - fy_rob * f_scale)
            v = int(self.pane_h / 2 + 0.3 * f_scale)
            bw = int(1.4 * f_scale)
            bh = int(1.8 * f_scale)

            # Draw 3D bounding box for forklift
            top_left = (max(10, u - bw // 2), max(10, v - bh))
            bot_right = (min(self.pane_w - 10, u + bw // 2), min(self.pane_h - 10, v))
            cv2.rectangle(canvas, top_left, bot_right, (0, 165, 255), 2)
            cv2.putText(canvas, f"FORKLIFT [{fx_rob:.1f}m]", (top_left[0], max(20, top_left[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)

        # Project Near-Ground Hazards in FPV
        for h in hazards:
            hx, hy, hz = h.centroid_3d
            if hx > 0.8:
                f_scale = 350.0 / hx
                u = int(self.pane_w / 2 - hy * f_scale)
                v = int(self.pane_h / 2 - (hz - 0.5) * f_scale)
                col = (0, 0, 255) if h.hazard_type == "NEGATIVE_DROP_OFF" else (0, 255, 255)
                cv2.circle(canvas, (u, v), 8, col, -1)
                cv2.putText(canvas, f"{h.hazard_type} [{hx:.1f}m]", (u - 40, v - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)

        # Header
        cv2.rectangle(canvas, (10, 10), (330, 42), (10, 15, 20), -1)
        cv2.putText(canvas, "ROBOT FPV: RGB-D CAMERA", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
        return canvas

    def _render_occupancy_mppi_view(self, occ_grid, mppi_res) -> np.ndarray:
        """Pane 3: 3D Neural Occupancy BEV Costmap & MPPI 10,000 Trajectory Fan."""
        canvas = np.zeros((self.pane_h, self.pane_w, 3), dtype=np.uint8)
        canvas[:] = (18, 20, 24)

        # Resize BEV costmap to fit pane
        costmap_resized = cv2.resize((occ_grid.bev_costmap * 255).astype(np.uint8), (self.pane_w - 40, self.pane_h - 80))
        costmap_color = cv2.applyColorMap(costmap_resized, cv2.COLORMAP_VIRIDIS)
        canvas[50:self.pane_h - 30, 20:self.pane_w - 20] = costmap_color

        # Overlay MPPI Trajectory Rollouts (Sample paths)
        # Coordinate mapping from robot frame [0..10m] x [-4..4m] to pane pixels
        def rob_to_pixel(x, y):
            px = int(20 + (x / 10.0) * (self.pane_w - 40))
            py = int((self.pane_h / 2) - (y / 4.0) * ((self.pane_h - 80) / 2))
            return px, py

        # Draw subset of 10,000 rollouts
        for path in mppi_res.all_rollouts[:40]:
            pts = [rob_to_pixel(pt[0], pt[1]) for pt in path]
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i+1], (60, 180, 80), 1)

        # Draw Optimal MPPI Path in Bright Gold
        opt_pts = [rob_to_pixel(pt[0], pt[1]) for pt in mppi_res.best_trajectory]
        for i in range(len(opt_pts) - 1):
            cv2.line(canvas, opt_pts[i], opt_pts[i+1], (0, 215, 255), 3)

        # Robot origin
        orig_pt = rob_to_pixel(0.0, 0.0)
        cv2.circle(canvas, orig_pt, 6, (0, 255, 0), -1)

        # Header & stats
        cv2.rectangle(canvas, (10, 10), (440, 42), (10, 15, 20), -1)
        cv2.putText(canvas, f"3D OCCUPANCY & MPPI (10k Rollouts - {mppi_res.latency_ms:.1f}ms)",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 2)
        return canvas

    def _render_vla_safety_cockpit(self, sim_time: float, prompt: str, vla_chunk, zone: SafetyZone, latency_ms: float) -> np.ndarray:
        """Pane 4: Live Neural VLA Attention Map & ISO 3691-4 Safety Telemetry."""
        canvas = np.zeros((self.pane_h, self.pane_w, 3), dtype=np.uint8)
        canvas[:] = (15, 18, 22)

        # Header
        cv2.rectangle(canvas, (10, 10), (380, 42), (10, 15, 20), -1)
        cv2.putText(canvas, "NEURAL VLA & ISO 3691-4 COCKPIT", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)

        # 1. Natural Language Mission Grounding
        y_off = 70
        cv2.putText(canvas, "TASK MISSION PROMPT:", (25, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        y_off += 22
        cv2.putText(canvas, f"\"{prompt[:55]}...\"", (25, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

        # 2. VLA Output State
        y_off += 35
        cv2.putText(canvas, f"VLA Intent: {vla_chunk.task_intent}", (25, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
        y_off += 22
        cv2.putText(canvas, f"VLA Confidence: {vla_chunk.confidence*100:.1f}%  |  Uncertainty sigma: {vla_chunk.uncertainty_sigma:.3f}",
                    (25, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # 3. Cross-Attention Heatmap
        y_off += 30
        cv2.putText(canvas, "CROSS-ATTENTION TOKEN WEIGHTS (Q: Lang, K: Vis):", (25, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
        y_off += 10
        attn_vis = cv2.resize((vla_chunk.attention_weights * 255).astype(np.uint8), (380, 45))
        attn_color = cv2.applyColorMap(attn_vis, cv2.COLORMAP_JET)
        canvas[y_off:y_off+45, 25:405] = attn_color

        # 4. ISO 3691-4 Safety Interlock Status
        y_off += 70
        zone_color = (0, 255, 0) if zone == SafetyZone.CLEAR else ((0, 255, 255) if zone == SafetyZone.WARNING else (0, 0, 255))
        cv2.rectangle(canvas, (25, y_off), (440, y_off + 45), (30, 35, 45), -1)
        cv2.rectangle(canvas, (25, y_off), (440, y_off + 45), zone_color, 2)
        cv2.putText(canvas, f"ISO 3691-4 FIELD: {zone.value}", (40, y_off + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, zone_color, 2)

        # 5. Telemetry & Robot Dynamics
        y_off += 70
        cv2.putText(canvas, f"Linear Velocity: {self.robot_vel[0]:.2f} m/s  |  Angular: {self.robot_vel[1]:.2f} rad/s",
                    (25, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        y_off += 22
        cv2.putText(canvas, f"Robot Position: ({self.robot_pos[0]:.2f}m, {self.robot_pos[1]:.2f}m)  |  Heading: {math.degrees(self.robot_pos[2]):.1f} deg",
                    (25, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        y_off += 22
        cv2.putText(canvas, f"VDA 5050 State: AUTOMATIC_DRIVING  |  Battery: 94.5%",
                    (25, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)

        return canvas


if __name__ == "__main__":
    out_path = os.path.join("C:\\Users\\Zeenat\\Desktop", "AURA_DRIVE_3D_DIGITAL_TWIN_SHOWCASE.mp4")
    twin = WarehouseDigitalTwin()
    twin.run_mission_simulation(out_path, duration_sec=12.0)

