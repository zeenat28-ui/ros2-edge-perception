"""
Embodied AI & Vision-Language-Action (VLA) Foundation Reasoning Node.
Grounds natural language mission instructions into 3D metric spatial goals
and actionable robot affordance primitives (PASS_LEFT, AVOID_HAZARD, DOCK_ALIGN).
"""

import re
import math
from typing import Dict, List, Optional, Tuple, Any

class VLAMissionAgent:
    """
    Multimodal Vision-Language-Action (VLA) Reasoning Engine.
    Parses unstructured text instructions and maps them into 3D spatial goals.
    """

    SUPPORTED_ACTIONS = ["NAVIGATE", "AVOID", "PASS_LEFT", "PASS_RIGHT", "STOP_HOLD", "DOCK", "INSPECT"]

    def __init__(self):
        self.current_mission_text = "Idle: Awaiting mission dispatch"
        self.active_action = "STOP_HOLD"
        self.spatial_target_coords: Optional[Tuple[float, float, float]] = None
        self.target_entity = ""
        self.confidence = 0.0

    def parse_instruction(self, instruction_text: str, active_tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Interprets natural language instruction, resolves spatial references,
        and grounds them against active 3D perceptual tracks.
        """
        self.current_mission_text = instruction_text
        text_lower = instruction_text.lower()

        # 1. Determine Intent / High-Level Action (Prioritize directional & safety actions over generic navigate)
        if "stop" in text_lower or "halt" in text_lower or "emergency" in text_lower:
            action = "STOP_HOLD"
        elif "pass left" in text_lower or ("left" in text_lower and ("overtake" in text_lower or "avoid" in text_lower or "bypass" in text_lower or "around" in text_lower)):
            action = "PASS_LEFT"
        elif "pass right" in text_lower or ("right" in text_lower and ("overtake" in text_lower or "avoid" in text_lower or "around" in text_lower)):
            action = "PASS_RIGHT"
        elif "dock" in text_lower or "park" in text_lower:
            action = "DOCK"
        elif "navigate" in text_lower or "cruise" in text_lower or "drive" in text_lower:
            action = "NAVIGATE"
        else:
            action = "AVOID"

        # 2. Entity Resolution (Target extraction)
        entities = ["car", "vehicle", "van", "truck", "human", "person", "pedestrian", "obstacle", "bay", "dock"]
        matched_entity = "obstacle"
        for ent in entities:
            if ent in text_lower:
                matched_entity = ent
                break

        # Semantic equivalence sets
        vehicle_set = {"car", "vehicle", "van", "truck", "automobile"}
        human_set = {"human", "person", "pedestrian"}

        # 3. Spatial Grounding against Active 3D Tracks
        target_track = None
        for track in active_tracks:
            t_class = track.get("class", "").lower()
            if (matched_entity in t_class or 
                (matched_entity in vehicle_set and t_class in vehicle_set) or
                (matched_entity in human_set and t_class in human_set) or
                matched_entity == "obstacle"):
                target_track = track
                break

        # If no specific track found, use default forward goal
        if target_track:
            tx = target_track.get("x", 0.0)
            ty = target_track.get("y", 0.0)
            tz = target_track.get("z", 20.0)
            
            # Compute affordance target offset based on action
            if action == "PASS_LEFT":
                goal_x = tx - 3.5  # 3.5m left corridor
                goal_z = tz + 15.0 # 15m beyond obstacle
            elif action == "PASS_RIGHT":
                goal_x = tx + 3.5
                goal_z = tz + 15.0
            elif action == "STOP_HOLD":
                goal_x = 0.0
                goal_z = max(5.0, tz - 12.0) # 12m safe buffer
            else:
                goal_x = 0.0
                goal_z = tz + 25.0
                
            confidence = 0.94
        else:
            goal_x = 0.0
            goal_z = 30.0
            confidence = 0.82

        self.active_action = action
        self.spatial_target_coords = (goal_x, 0.0, goal_z)
        self.target_entity = matched_entity
        self.confidence = confidence

        return {
            "mission_text": instruction_text,
            "action": action,
            "target_entity": matched_entity,
            "spatial_goal_3d": [goal_x, 0.0, goal_z],
            "confidence": confidence,
            "reasoning": f"Grounded '{matched_entity}' to 3D waypoint [{goal_x:.1f}m, 0.0m, {goal_z:.1f}m] with affordance {action}."
        }
