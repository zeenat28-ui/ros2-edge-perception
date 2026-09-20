"""
Structured Safety Interface & Deterministic Safety Request Contracts for AMRs.

Perception nodes emit supervisory safety requests over ROS 2 topics (/perception/safety_state
and /perception/safety_alert) to notify external safety supervisors or robot controllers.
Perception does NOT act as a standalone certified emergency stop; it issues versioned
safety requests with explicit reason codes and recommended actions.
"""

import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Any


class SafetyState(str, Enum):
    """Operational safety states of the perception node."""
    STARTUP = "STARTUP"
    NOMINAL = "NOMINAL"
    DEGRADED = "DEGRADED"
    STOP_REQUESTED = "STOP_REQUESTED"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class SafetyReasonCode(str, Enum):
    """Explicit deterministic reason codes for safety state transitions."""
    NONE = "NONE"
    CAMERA_TIMEOUT = "CAMERA_TIMEOUT"
    DEPTH_TIMEOUT = "DEPTH_TIMEOUT"
    LIDAR_TIMEOUT = "LIDAR_TIMEOUT"
    TTC_LIMIT = "TTC_LIMIT"
    INVALID_DEPTH = "INVALID_DEPTH"
    INTRINSICS_INVALID = "INTRINSICS_INVALID"
    INFERENCE_FAILURE = "INFERENCE_FAILURE"
    STALE_FRAME = "STALE_FRAME"


class RecommendedAction(str, Enum):
    """Recommended actions requested to the robot safety supervisor / Nav2 controller."""
    CONTINUE = "CONTINUE"
    DECELERATE = "DECELERATE"
    STOP = "STOP"
    MANUAL_INTERVENTION = "MANUAL_INTERVENTION"


@dataclass
class SafetyRequest:
    """Versioned structured safety request payload."""
    state: SafetyState
    reason_code: SafetyReasonCode
    timestamp: float
    source: str = "ros2_edge_perception"
    recommended_action: RecommendedAction = RecommendedAction.CONTINUE
    ttc_seconds: Optional[float] = None
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value if isinstance(self.state, SafetyState) else str(self.state),
            "reason_code": self.reason_code.value if isinstance(self.reason_code, SafetyReasonCode) else str(self.reason_code),
            "timestamp": round(self.timestamp, 4),
            "source": self.source,
            "recommended_action": self.recommended_action.value if isinstance(self.recommended_action, RecommendedAction) else str(self.recommended_action),
            "ttc_seconds": round(self.ttc_seconds, 3) if self.ttc_seconds is not None else None,
            "details": self.details,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "SafetyRequest":
        data = json.loads(json_str)
        return cls(
            state=SafetyState(data.get("state", "NOMINAL")),
            reason_code=SafetyReasonCode(data.get("reason_code", "NONE")),
            timestamp=float(data.get("timestamp", time.time())),
            source=data.get("source", "ros2_edge_perception"),
            recommended_action=RecommendedAction(data.get("recommended_action", "CONTINUE")),
            ttc_seconds=data.get("ttc_seconds"),
            details=data.get("details", ""),
        )
