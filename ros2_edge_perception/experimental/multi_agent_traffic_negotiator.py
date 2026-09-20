#!/usr/bin/env python3
"""
Multi-Agent Cooperative Right-of-Way Traffic Negotiator.

Resolves narrow aisle deadlocks between opposing mobile robots:
- Head-on oncoming peer detection using forward detection horizons
- Deterministic right-of-way arbitration: Payload status (laden > unladen) and priority tiers
- Cooperative avoidance maneuvers: Dominant robot maintains corridor; yielding robot hugs rack wall
"""

import time
import math
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


class TrafficPriorityLevel(Enum):
    EMERGENCY_INTERVENTION = 100
    LOADED_PALLET_TRANSPORT = 75
    UNLADEN_REPOSITIONING = 50
    IDLE_STANDBY = 25


class AgentTrafficRole(Enum):
    DOMINANT_RIGHT_OF_WAY = "DOMINANT_RIGHT_OF_WAY"
    YIELDING_PULL_ASIDE = "YIELDING_PULL_ASIDE"
    YIELDING_PASSING_BAY = "YIELDING_PASSING_BAY"
    CLEAR_CORRIDOR = "CLEAR_CORRIDOR"


@dataclass
class PeerAMRState:
    """Telemetry broadcast of an oncoming peer robot."""
    robot_id: str
    position_xyz: Tuple[float, float, float]
    velocity_v_omega: Tuple[float, float]
    heading_rad: float
    corridor_id: str
    priority: TrafficPriorityLevel
    is_loaded: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrafficNegotiationDecision:
    """Cooperative action decision for the local robot."""
    role: AgentTrafficRole
    target_lane_offset_m: float       # Lateral shift (+left, -right) to clear aisle
    speed_governor_mps: float          # Safe maximum speed during pass
    should_hold_position: bool
    peer_robot_id: Optional[str]
    deadlock_prevented: bool
    explanation: str
    timestamp: float = field(default_factory=time.time)


class MultiAgentTrafficNegotiator:
    """
    Cooperative traffic negotiator resolving narrow corridor bottlenecks.
    Enforces deterministic Right-of-Way rules to guarantee deadlock-free flow.
    """

    def __init__(
        self,
        local_robot_id: str = "AURA-AMR-001",
        aisle_width_m: float = 1.70,     # Narrow aisle standard width
        robot_width_m: float = 0.80,
        lateral_clearance_m: float = 0.15,
        forward_detection_range_m: float = 8.0
    ):
        self.local_robot_id = local_robot_id
        self.aisle_width = aisle_width_m
        self.robot_width = robot_width_m
        self.lateral_clearance = lateral_clearance_m
        self.forward_detection_range = forward_detection_range_m

        self.local_priority = TrafficPriorityLevel.UNLADEN_REPOSITIONING
        self.local_is_loaded = False
        self.total_deadlocks_resolved = 0

    def set_local_status(self, is_loaded: bool, priority: Optional[TrafficPriorityLevel] = None):
        """Updates local robot payload and operational priority."""
        self.local_is_loaded = is_loaded
        if priority:
            self.local_priority = priority
        else:
            self.local_priority = TrafficPriorityLevel.LOADED_PALLET_TRANSPORT if is_loaded else TrafficPriorityLevel.UNLADEN_REPOSITIONING

    def negotiate_traffic(
        self,
        local_pose: Tuple[float, float, float],    # (x, y, theta)
        local_corridor_id: str,
        peer_robots: List[PeerAMRState]
    ) -> TrafficNegotiationDecision:
        """
        Evaluates peer robots in corridor and resolves right-of-way.
        """
        lx, ly, ltheta = local_pose

        # 1. Filter for peers in the same corridor approaching head-on
        head_on_peer: Optional[PeerAMRState] = None
        min_peer_dist = 999.0

        for peer in peer_robots:
            if peer.robot_id == self.local_robot_id:
                continue
            if peer.corridor_id != local_corridor_id:
                continue

            px, py, _ = peer.position_xyz
            dx = px - lx
            dy = py - ly
            dist = math.sqrt(dx**2 + dy**2)

            if dist > self.forward_detection_range:
                continue

            # Check if oncoming (headings roughly opposite: dot product of forward vectors < 0)
            dot_headings = math.cos(ltheta) * math.cos(peer.heading_rad) + math.sin(ltheta) * math.sin(peer.heading_rad)
            if dot_headings < -0.3:  # Facing towards each other
                if dist < min_peer_dist:
                    min_peer_dist = dist
                    head_on_peer = peer

        # If no oncoming peer, corridor is clear
        if head_on_peer is None:
            return TrafficNegotiationDecision(
                role=AgentTrafficRole.CLEAR_CORRIDOR,
                target_lane_offset_m=0.0,
                speed_governor_mps=1.5,
                should_hold_position=False,
                peer_robot_id=None,
                deadlock_prevented=False,
                explanation="No oncoming peer detected in corridor."
            )

        # 2. Right-of-Way Arbitration
        # Rule A: Priority Comparison (Loaded > Unladen)
        local_p = self.local_priority.value
        peer_p = head_on_peer.priority.value

        if local_p > peer_p:
            local_has_right_of_way = True
            reason = f"Local priority ({self.local_priority.name}) > Peer ({head_on_peer.priority.name})"
        elif local_p < peer_p:
            local_has_right_of_way = False
            reason = f"Peer priority ({head_on_peer.priority.name}) > Local ({self.local_priority.name})"
        else:
            # Rule B: Tie-breaker by Lexicographical Robot ID
            local_has_right_of_way = self.local_robot_id < head_on_peer.robot_id
            reason = f"Priority tie; resolved by ID arbitration ({self.local_robot_id} vs {head_on_peer.robot_id})"

        self.total_deadlocks_resolved += 1

        # 3. Generate Cooperative Decision
        if local_has_right_of_way:
            # Dominant robot maintains center-left right-of-way
            return TrafficNegotiationDecision(
                role=AgentTrafficRole.DOMINANT_RIGHT_OF_WAY,
                target_lane_offset_m=0.15,  # Slight left bias
                speed_governor_mps=1.0,     # Governed passing speed
                should_hold_position=False,
                peer_robot_id=head_on_peer.robot_id,
                deadlock_prevented=True,
                explanation=f"DOMINANT: {reason}. Maintaining passage."
            )
        else:
            # Yielding robot hugs right wall (-Y lateral shift)
            # Required passing clearance: robot_width + clearance
            wall_hug_offset = -0.38  # 38cm right lateral shift towards rack
            hold = min_peer_dist < 2.2  # Hold position if very close

            return TrafficNegotiationDecision(
                role=AgentTrafficRole.YIELDING_PULL_ASIDE,
                target_lane_offset_m=wall_hug_offset,
                speed_governor_mps=0.35,  # Slow controlled crawl
                should_hold_position=hold,
                peer_robot_id=head_on_peer.robot_id,
                deadlock_prevented=True,
                explanation=f"YIELDING: {reason}. Hugging right wall at {wall_hug_offset}m."
            )

