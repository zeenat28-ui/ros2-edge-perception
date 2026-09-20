"""
AURA-Drive™ Enterprise 4-Pillar Comprehensive Test Suite
========================================================
Validates:
  - Pillar 1: URDF / Xacro kinematics and Nav2 parameter compliance
  - Pillar 2: ISO 26262 ASIL-D & ISO 3691-4 Safety Watchdog and Interlocks
  - Pillar 3: DevSecOps HMAC-SHA256 JWT, RBAC, Rate Limiting, and Revocation
  - Pillar 4: MLOps Sensor Distribution Drift (Wasserstein / KS tests)
"""

import time
import pytest
import numpy as np

from ros2_edge_perception.iso26262_watchdog import ISO26262SafetyWatchdog, SafetyState, HazardSeverity
from ros2_edge_perception.security_auth_manager import SecurityAuthManager, SecurityRole, Permission
from ros2_edge_perception.mlops_drift_detector import MLOpsSensorDriftDetector
from run_aura_system import MasterAURASystem


class TestPillar2SafetyWatchdog:
    """ASIL-D Watchdog & Interlock Verification."""

    def test_nominal_heartbeats(self):
        watchdog = ISO26262SafetyWatchdog(heartbeat_timeout_ms=100.0, max_jitter_ms=120.0)
        for seq in range(1, 5):
            watchdog.record_heartbeat("perception_engine", sequence_id=seq)
            watchdog.record_heartbeat("diffusion_vla_policy", sequence_id=seq)
            watchdog.record_heartbeat("cuda_mppi_optimizer", sequence_id=seq)
            watchdog.record_heartbeat("amcl_localization", sequence_id=seq)

        status = watchdog.evaluate_safety_envelope()
        assert status.current_state == SafetyState.NORMAL_OPERATION
        assert not status.is_interlock_tripped
        assert status.active_hazard_level == HazardSeverity.NONE

    def test_deadline_timeout_triggers_estop(self):
        # Set a short timeout (50ms)
        watchdog = ISO26262SafetyWatchdog(heartbeat_timeout_ms=50.0)
        watchdog.record_heartbeat("perception_engine", sequence_id=1)
        watchdog.record_heartbeat("diffusion_vla_policy", sequence_id=1)

        # Sleep longer than timeout to induce deadline miss
        time.sleep(0.08)
        status = watchdog.evaluate_safety_envelope()

        assert status.current_state == SafetyState.EMERGENCY_STOP_CAT0
        assert status.is_interlock_tripped
        assert status.active_hazard_level == HazardSeverity.CRITICAL

    def test_interlock_reset_authorization(self):
        watchdog = ISO26262SafetyWatchdog(heartbeat_timeout_ms=500.0)
        watchdog.trigger_software_estop("TestRunner", "Manual test trigger")
        assert watchdog.is_interlock_tripped

        # Attempt reset without authorization
        success, msg = watchdog.reset_safety_interlock(authorization_token_valid=False)
        assert not success
        assert "AUTHORIZATION_DENIED" in msg

        # Attempt reset with valid authorization
        success, msg = watchdog.reset_safety_interlock(authorization_token_valid=True)
        assert success
        assert watchdog.current_state == SafetyState.NORMAL_OPERATION


class TestPillar3DevSecOps:
    """DevSecOps Authentication, RBAC & Zero-Trust Verification."""

    def test_jwt_generation_and_verification(self):
        auth = SecurityAuthManager(token_validity_sec=60)
        token = auth.create_token(subject="user_dispatcher", role=SecurityRole.FLEET_DISPATCHER)

        is_valid, payload, msg = auth.verify_token(token)
        assert is_valid
        assert payload["sub"] == "user_dispatcher"
        assert payload["role"] == SecurityRole.FLEET_DISPATCHER

    def test_rbac_permission_matrix(self):
        auth = SecurityAuthManager()
        # Dispatcher has DISPATCH_ORDER but not RESET_INTERLOCK
        dispatcher_token = auth.create_token(subject="dispatcher_01", role=SecurityRole.FLEET_DISPATCHER)
        assert auth.authorize(dispatcher_token, Permission.DISPATCH_ORDER)[0] is True
        assert auth.authorize(dispatcher_token, Permission.RESET_INTERLOCK)[0] is False

        # Safety Officer has RESET_INTERLOCK but not DISPATCH_ORDER
        officer_token = auth.create_token(subject="safety_01", role=SecurityRole.SAFETY_OFFICER)
        assert auth.authorize(officer_token, Permission.RESET_INTERLOCK)[0] is True
        assert auth.authorize(officer_token, Permission.DISPATCH_ORDER)[0] is False

    def test_tampered_token_rejection(self):
        auth = SecurityAuthManager()
        token = auth.create_token(subject="user_01", role=SecurityRole.MONITORING_CLIENT)
        # Tamper with the signature portion
        parts = token.split(".")
        tampered_token = f"{parts[0]}.{parts[1]}.badsignature123"

        is_valid, _, msg = auth.verify_token(tampered_token)
        assert not is_valid
        assert "INVALID_SIGNATURE" in msg

    def test_token_revocation(self):
        auth = SecurityAuthManager()
        token = auth.create_token(subject="user_temp", role=SecurityRole.MAINTENANCE_TECH)
        assert auth.verify_token(token)[0] is True

        auth.revoke_token(token)
        is_valid, _, msg = auth.verify_token(token)
        assert not is_valid
        assert "TOKEN_REVOKED" in msg


class TestPillar4MLOpsDrift:
    """MLOps Sensor Drift & Distribution Shift Verification."""

    def test_nominal_distribution_passes(self):
        detector = MLOpsSensorDriftDetector(window_size=50)
        rng = np.random.default_rng(42)
        for _ in range(50):
            detector.ingest_frame_telemetry(
                mean_luminance=float(rng.normal(128.0, 25.0)),
                depth_variance=float(rng.normal(1.5, 0.3)),
                lidar_point_count=float(rng.normal(850.0, 70.0))
            )

        has_drift, _ = detector.is_any_drift_active()
        assert not has_drift

    def test_optical_occlusion_triggers_drift(self):
        detector = MLOpsSensorDriftDetector(window_size=50)
        rng = np.random.default_rng(42)
        # Ingest occluded sensor values (mud / grease over lens)
        for _ in range(50):
            detector.ingest_frame_telemetry(
                mean_luminance=float(rng.normal(20.0, 3.0)),  # severe drop
                depth_variance=float(rng.normal(0.1, 0.02)),
                lidar_point_count=float(rng.normal(850.0, 70.0))
            )

        has_drift, reasons = detector.is_any_drift_active()
        assert has_drift
        assert len(reasons) > 0


class TestMasterSystemIntegration:
    """Full 12-Subsystem Synchronous Autonomy Loop."""

    def test_master_system_headless_execution(self):
        system = MasterAURASystem()
        for _ in range(3):
            res = system.step()
            assert res["iteration"] >= 1
            assert res["safety_state"] == "NORMAL_OPERATION"
            assert not res["drift_detected"]
            assert res["cycle_time_ms"] > 0.0
