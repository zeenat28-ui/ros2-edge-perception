"""
Unit Tests for Embodied AI & Vision-Language-Action (VLA) Reasoning Agent.
"""

import pytest
from ros2_edge_perception.vla_agent_node import VLAMissionAgent

def test_vla_pass_left_instruction():
    agent = VLAMissionAgent()
    tracks = [{"id": 1, "class": "car", "x": 0.0, "y": 0.0, "z": 20.0}]
    
    result = agent.parse_instruction("Navigate around the stalled car on the left", tracks)
    assert result["action"] == "PASS_LEFT"
    assert result["target_entity"] == "car"
    assert result["spatial_goal_3d"][0] == -3.5 # 3.5m left corridor
    assert result["spatial_goal_3d"][2] == 35.0 # 15m beyond obstacle
    assert result["confidence"] > 0.9

def test_vla_emergency_stop_instruction():
    agent = VLAMissionAgent()
    tracks = [{"id": 2, "class": "human", "x": 0.5, "y": 0.0, "z": 18.0}]
    
    result = agent.parse_instruction("Emergency stop for the person ahead", tracks)
    assert result["action"] == "STOP_HOLD"
    assert result["target_entity"] == "person"
    assert result["spatial_goal_3d"][2] == 6.0 # Safe 12m buffer (18 - 12)

def test_vla_docking_instruction():
    agent = VLAMissionAgent()
    tracks = []
    
    result = agent.parse_instruction("Dock at charging bay", tracks)
    assert result["action"] == "DOCK"
    assert result["target_entity"] in ["bay", "dock"]
