#!/usr/bin/env python3
"""
=============================================================================
AURA-DRIVE 2026: ENTERPRISE PRODUCTION HARDENING TEST SUITE
=============================================================================
Comprehensive unit and integration test suite validating the 5 enterprise
enhancements required for Tier-1 AMR procurement:
1. Temporal De-Ghosting & Specular Glare Rejection
2. Sub-Millimeter Floor Tag (AprilTag/QR) Relocalization
3. VDA 5050 Multi-Agent Traffic & Deadlock Negotiation
4. SAP EWM / Manhattan WMS REST API Gateway
5. NVIDIA Jetson AGX Orin Edge Power & Thermal Profiler
=============================================================================
"""

import math
import pytest
import numpy as np
from fastapi.testclient import TestClient

from ros2_edge_perception.deghosting_hazard_filter import (
    TemporalDeghostingFilter,
    TrackedHazardCandidate
)
from ros2_edge_perception.floor_tag_relocalization import (
    FloorTagRelocalizer,
    FloorTagGlobalPose
)
from ros2_edge_perception.multi_agent_traffic_negotiator import (
    MultiAgentTrafficNegotiator,
    PeerAMRState,
    TrafficPriorityLevel,
    AgentTrafficRole
)
from ros2_edge_perception.wms_rest_gateway import (
    EnterpriseWMSGateway,
    TransportOrderRequest,
    EmergencyStopRequest,
    InterlockResetRequest
)
from ros2_edge_perception.jetson_edge_profiler import (
    JetsonEdgeProfiler,
    EdgeHardwareMetrics
)


# ===========================================================================
# 1. Tests for Temporal De-Ghosting & Glare Filter
# ===========================================================================

def test_deghosting_rejects_single_frame_glare():
    """Validates that a 1-frame specular reflection or dust burst is rejected."""
    filter_engine = TemporalDeghostingFilter(window_size=5, min_persistence_threshold=0.70)

    # 1st frame: Glare artifact appears
    glare_candidates = [{
        "hazard_type": "SPECULAR_GLARE",
        "centroid_3d": [2.5, 0.1, 0.05],
        "confidence": 0.85
    }]
    res1 = filter_engine.filter_hazards(glare_candidates)

    # Must be rejected on first frame (persistence < 0.70)
    assert len(res1.verified_hazards) == 0
    assert len(res1.rejected_ghosts) == 1
    assert res1.rejected_ghosts[0]["rejection_reason"] == "TRANSIENT_NOISE_OR_GLARE"
    assert res1.phantom_stops_prevented >= 1


def test_deghosting_verifies_persistent_hazard():
    """Validates that a persistent physical hazard (e.g. cable on floor) is confirmed."""
    filter_engine = TemporalDeghostingFilter(window_size=5, min_persistence_threshold=0.70)

    # Physical cable candidate present over 5 consecutive frames
    cable_candidate = {
        "hazard_type": "FLOOR_CABLE",
        "centroid_3d": [3.0, 0.0, 0.02],
        "confidence": 0.90
    }

    verified_found = False
    for frame in range(5):
        res = filter_engine.filter_hazards([cable_candidate.copy()])
        if len(res.verified_hazards) > 0:
            verified_found = True
            break

    assert verified_found is True
    assert res.verified_hazards[0]["verified"] is True
    assert res.verified_hazards[0]["persistence_score"] >= 0.70


def test_deghosting_dock_dropoff_early_trigger():
    """Validates that negative drop-offs (loading docks) trigger earlier for safety."""
    filter_engine = TemporalDeghostingFilter(window_size=5, min_persistence_threshold=0.70)

    dock_candidate = {
        "hazard_type": "NEGATIVE_DROP_OFF",
        "centroid_3d": [4.0, 0.0, -0.45],
        "confidence": 0.95
    }

    # Run 2 frames
    filter_engine.filter_hazards([dock_candidate.copy()])
    res = filter_engine.filter_hazards([dock_candidate.copy()])

    # With lowered threshold for dock drop-offs (0.55), it should verify promptly
    assert len(res.verified_hazards) == 1
    assert res.verified_hazards[0]["hazard_type"] == "NEGATIVE_DROP_OFF"


# ===========================================================================
# 2. Tests for Floor Tag (AprilTag/QR) Relocalization
# ===========================================================================

def test_floor_tag_map_initialization():
    """Verifies that warehouse floor tag dictionary is properly mapped."""
    relocalizer = FloorTagRelocalizer()
    assert len(relocalizer.floor_map) == 21  # 3 aisles * 7 bays
    assert 101 in relocalizer.floor_map  # Aisle 1, Bay 1
    assert relocalizer.floor_map[101].aisle_id == "AISLE-1"
    assert relocalizer.floor_map[101].global_x == 2.0


def test_floor_tag_pnp_sub_millimeter_accuracy():
    """Verifies solvePnP relocalization computes sub-2mm accuracy and resets drift."""
    relocalizer = FloorTagRelocalizer(camera_height_m=0.35, tag_size_m=0.12)

    # Project corners for a tag directly under the camera (tag 102: x=6.0, y=-4.0)
    # 12cm tag at 35cm distance: half-size s=0.06m
    # fx=500, cx=320, cy=240
    # projected width: 0.12 * 500 / 0.35 = ~171.4 pixels
    s = 0.06
    z = 0.35
    fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0

    corners = np.array([
        [cx - (s * fx / z), cy - (s * fy / z)],
        [cx + (s * fx / z), cy - (s * fy / z)],
        [cx + (s * fx / z), cy + (s * fy / z)],
        [cx - (s * fx / z), cy + (s * fy / z)]
    ], dtype=np.float32)

    # Robot odometry has accumulated 0.35m drift in X:
    raw_odometry = (6.35, -4.0, 0.0)
    result = relocalizer.estimate_pose_from_corners(102, corners, raw_odometry)

    assert result is not None
    assert result.relocalized is True
    # Corrected global X should be ~6.0m (within 5mm)
    assert abs(result.estimated_global_pose[0] - 6.0) < 0.005
    # Drift correction should identify the ~ -0.35m error
    assert abs(result.drift_correction[0] - (-0.35)) < 0.005
    assert result.accuracy_margin_mm < 2.5


def test_floor_tag_unknown_id():
    """Verifies that unknown tag ID returns None without crashing."""
    relocalizer = FloorTagRelocalizer()
    corners = np.zeros((4, 2), dtype=np.float32)
    res = relocalizer.estimate_pose_from_corners(99999, corners, (0.0, 0.0, 0.0))
    assert res is None


# ===========================================================================
# 3. Tests for Multi-Agent Traffic & Deadlock Negotiator
# ===========================================================================

def test_multi_agent_clear_corridor():
    """Verifies normal corridor operation when no peer robots are approaching."""
    negotiator = MultiAgentTrafficNegotiator(local_robot_id="AURA-AMR-001")
    decision = negotiator.negotiate_traffic(
        local_pose=(2.0, 0.0, 0.0),
        local_corridor_id="CORRIDOR-A",
        peer_robots=[]
    )
    assert decision.role == AgentTrafficRole.CLEAR_CORRIDOR
    assert decision.speed_governor_mps == 1.5
    assert decision.deadlock_prevented is False


def test_multi_agent_loaded_vs_unladen_precedence():
    """Verifies that a loaded pallet AMR takes precedence over an unladen AMR in a narrow aisle."""
    negotiator = MultiAgentTrafficNegotiator(local_robot_id="AURA-AMR-001")
    # Local robot is carrying a 600kg pallet
    negotiator.set_local_status(is_loaded=True, priority=TrafficPriorityLevel.LOADED_PALLET_TRANSPORT)

    # Peer robot is approaching head-on (heading = pi, local heading = 0) and is unladen
    peer = PeerAMRState(
        robot_id="AURA-AMR-002",
        position_xyz=(6.0, 0.0, 0.0),
        velocity_v_omega=(0.8, 0.0),
        heading_rad=math.pi,
        corridor_id="CORRIDOR-A",
        priority=TrafficPriorityLevel.UNLADEN_REPOSITIONING,
        is_loaded=False
    )

    decision = negotiator.negotiate_traffic(
        local_pose=(2.0, 0.0, 0.0),
        local_corridor_id="CORRIDOR-A",
        peer_robots=[peer]
    )

    assert decision.role == AgentTrafficRole.DOMINANT_RIGHT_OF_WAY
    assert decision.speed_governor_mps == 1.0
    assert decision.deadlock_prevented is True
    assert decision.peer_robot_id == "AURA-AMR-002"


def test_multi_agent_unladen_yields_to_loaded():
    """Verifies that an unladen AMR yields and hugs the right wall (-0.38m)."""
    negotiator = MultiAgentTrafficNegotiator(local_robot_id="AURA-AMR-001")
    # Local robot is unladen
    negotiator.set_local_status(is_loaded=False, priority=TrafficPriorityLevel.UNLADEN_REPOSITIONING)

    # Peer robot is carrying a loaded pallet
    peer = PeerAMRState(
        robot_id="AURA-AMR-002",
        position_xyz=(5.0, 0.0, 0.0),
        velocity_v_omega=(1.0, 0.0),
        heading_rad=math.pi,
        corridor_id="CORRIDOR-A",
        priority=TrafficPriorityLevel.LOADED_PALLET_TRANSPORT,
        is_loaded=True
    )

    decision = negotiator.negotiate_traffic(
        local_pose=(2.0, 0.0, 0.0),
        local_corridor_id="CORRIDOR-A",
        peer_robots=[peer]
    )

    assert decision.role == AgentTrafficRole.YIELDING_PULL_ASIDE
    assert decision.target_lane_offset_m == -0.38
    assert decision.speed_governor_mps == 0.35
    assert decision.deadlock_prevented is True


# ===========================================================================
# 4. Tests for Enterprise WMS / ERP REST API Gateway
# ===========================================================================

def test_wms_gateway_health_and_fleet_status():
    """Verifies WMS healthz and fleet telemetry endpoints."""
    gateway = EnterpriseWMSGateway()
    client = TestClient(gateway.app)

    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json()["status"] == "HEALTHY"

    res_fleet = client.get("/api/v1/fleet/status")
    assert res_fleet.status_code == 200
    data = res_fleet.json()
    assert data["total_robots"] == 3
    assert data["emergency_stop_active"] is False


def test_wms_gateway_order_dispatch_lifecycle():
    """Verifies SAP EWM transport order dispatch and assignment."""
    gateway = EnterpriseWMSGateway()
    client = TestClient(gateway.app)

    order_payload = {
        "source_bay": "AISLE-1-BAY-03",
        "destination_bay": "DOCK-OUTBOUND-01",
        "pallet_id": "PAL-99214",
        "priority": 80,
        "weight_kg": 520.0
    }

    res = client.post("/api/v1/orders/dispatch", json=order_payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ACCEPTED"
    assert body["assigned_robot_id"] in ["AURA-AMR-001", "AURA-AMR-002"]
    assert body["estimated_transit_time_sec"] > 0.0


def test_wms_gateway_emergency_stop_and_reset():
    """Verifies facility E-Stop triggers interlock and supervisor reset clears it."""
    gateway = EnterpriseWMSGateway()
    client = TestClient(gateway.app)

    # 1. Trigger E-Stop
    estop_payload = {
        "initiator": "SAP_SAFETY_SUPERVISOR",
        "reason": "Fire alarm zone 3",
        "emergency_code": "ESTOP_LEVEL_1_FACILITY"
    }
    res_estop = client.post("/api/v1/fleet/emergency_stop", json=estop_payload)
    assert res_estop.status_code == 200
    assert res_estop.json()["status"] == "ESTOP_TRIGGERED"

    # Orders should now be rejected (503)
    res_order = client.post("/api/v1/orders/dispatch", json={
        "source_bay": "AISLE-2-BAY-01",
        "destination_bay": "DOCK-02"
    })
    assert res_order.status_code == 503

    # 2. Attempt reset with wrong PIN (403)
    bad_reset = client.post("/api/v1/fleet/reset_interlock", json={
        "supervisor_pin": "WRONG-PIN",
        "clearance_token": "CLR-12345",
        "reason": "Test clearance"
    })
    assert bad_reset.status_code == 403

    # 3. Reset with valid PIN (200)
    good_reset = client.post("/api/v1/fleet/reset_interlock", json={
        "supervisor_pin": "SAP-SUPERVISOR-99",
        "clearance_token": "CLR-88492-AUTH",
        "reason": "Zone 3 cleared by safety marshal"
    })
    assert good_reset.status_code == 200
    assert good_reset.json()["status"] == "INTERLOCK_CLEARED"


# ===========================================================================
# 5. Tests for NVIDIA Jetson Edge Power & Thermal Profiler
# ===========================================================================

def test_jetson_profiler_compliance():
    """Verifies edge hardware profiler certifies compliance within AMR constraints."""
    profiler = JetsonEdgeProfiler(
        target_power_budget_w=45.0,
        max_thermal_limit_c=65.0,
        max_memory_limit_mb=2048.0
    )

    for load in [0.5, 1.0, 1.5]:
        metric = profiler.sample_metrics(pipeline_load_factor=load)
        assert metric.power_budget_compliant is True
        assert metric.thermal_compliant is True
        assert metric.memory_compliant is True
        assert metric.total_power_watts <= 45.0
        assert metric.gpu_temperature_c <= 65.0

    report = profiler.generate_procurement_report()
    assert report["certified_for_tier1_amr"] is True
    assert report["power_audit"]["compliant"] is True
    assert report["thermal_audit"]["compliant"] is True
    assert report["memory_audit"]["compliant"] is True
    assert report["power_audit"]["power_margin_w"] > 0.0

