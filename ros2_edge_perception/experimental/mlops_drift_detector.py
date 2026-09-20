"""
Statistical Sensor Distribution Drift and Out-of-Distribution (OOD) Detector.

Monitors multi-modal camera and LiDAR streams for operational distribution shifts:
- 1D 2-Wasserstein Distance (Earth Mover's Distance)
- Two-sample Kolmogorov-Smirnov (KS) hypothesis testing
- Detects optical occlusion (lens grease/dust), lighting shifts, and sensor degradation
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [MLOps Drift] %(message)s")
logger = logging.getLogger("AURA_MLOpsDrift")


def compute_wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    """Computes 1D Wasserstein distance between two empirical distributions."""
    if HAS_SCIPY:
        return float(stats.wasserstein_distance(u, v))
    qs = np.linspace(0.01, 0.99, 100)
    u_q = np.quantile(u, qs)
    v_q = np.quantile(v, qs)
    return float(np.mean(np.abs(u_q - v_q)))


def compute_ks_2samp(u: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    """Computes two-sample Kolmogorov-Smirnov statistic and asymptotic p-value."""
    if HAS_SCIPY:
        res = stats.ks_2samp(u, v)
        return float(res.statistic), float(res.pvalue)

    n1 = len(u)
    n2 = len(v)
    data_all = np.sort(np.concatenate([u, v]))
    cdf1 = np.searchsorted(np.sort(u), data_all, side='right') / n1
    cdf2 = np.searchsorted(np.sort(v), data_all, side='right') / n2
    d = float(np.max(np.abs(cdf1 - cdf2)))

    n_eff = np.sqrt((n1 * n2) / (n1 + n2))
    z = (n_eff + 0.12 + 0.11 / n_eff) * d
    terms = np.array([(-1.0)**(j - 1) * np.exp(-2.0 * (j**2) * (z**2)) for j in range(1, 25)])
    p_val = float(np.clip(2.0 * np.sum(terms), 0.0, 1.0))
    return d, p_val


@dataclass
class DriftMetricResult:
    modality: str
    wasserstein_dist: float
    ks_statistic: float
    p_value: float
    drift_detected: bool
    severity: float  # 0.0 to 1.0
    recommended_action: str


@dataclass
class SensorBaseline:
    luminance_sample: np.ndarray
    depth_variance_sample: np.ndarray
    lidar_density_sample: np.ndarray


class MLOpsSensorDriftDetector:
    """
    Online statistical distribution shift monitor for camera & LiDAR data streams.
    """

    def __init__(
        self,
        window_size: int = 100,
        wasserstein_threshold: float = 0.15,
        ks_alpha: float = 0.05
    ):
        self.window_size = window_size
        self.wasserstein_threshold = wasserstein_threshold
        self.ks_alpha = ks_alpha

        # Rolling buffers for online sliding window
        self._curr_luminance: List[float] = []
        self._curr_depth_var: List[float] = []
        self._curr_lidar_dens: List[float] = []

        # Reference golden baseline distributions
        self._baseline: Optional[SensorBaseline] = None
        self._is_calibrated = False

        # Initialize standard baseline distributions
        self._initialize_default_baseline()

    def _initialize_default_baseline(self):
        """Synthesize verified nominal operational baselines (clean warehouse conditions)."""
        rng = np.random.default_rng(seed=42)
        # Nominal clean camera luminance: mean ~128, std ~30
        lum_base = rng.normal(loc=128.0, scale=30.0, size=500).clip(0, 255)
        # Nominal depth variance across scene: mean ~1.5m, std ~0.4m
        depth_base = rng.normal(loc=1.5, scale=0.4, size=500).clip(0.1, 10.0)
        # Nominal LiDAR point cloud density: mean ~850 pts/scan, std ~80
        lidar_base = rng.normal(loc=850.0, scale=80.0, size=500).clip(100, 2000)

        self._baseline = SensorBaseline(
            luminance_sample=lum_base,
            depth_variance_sample=depth_base,
            lidar_density_sample=lidar_base
        )
        self._is_calibrated = True

    def set_custom_baseline(self, luminance: np.ndarray, depth_var: np.ndarray, lidar_dens: np.ndarray):
        """Inject calibrated physical environment baseline data."""
        self._baseline = SensorBaseline(
            luminance_sample=np.asarray(luminance, dtype=np.float32),
            depth_variance_sample=np.asarray(depth_var, dtype=np.float32),
            lidar_density_sample=np.asarray(lidar_dens, dtype=np.float32)
        )
        self._is_calibrated = True
        logger.info("Custom sensor distribution baseline calibrated successfully.")

    def ingest_frame_telemetry(self, mean_luminance: float, depth_variance: float, lidar_point_count: float):
        """Ingest live single-step sensor telemetry into rolling evaluation window."""
        self._curr_luminance.append(float(mean_luminance))
        self._curr_depth_var.append(float(depth_variance))
        self._curr_lidar_dens.append(float(lidar_point_count))

        # Maintain bounded sliding window
        if len(self._curr_luminance) > self.window_size:
            self._curr_luminance.pop(0)
            self._curr_depth_var.pop(0)
            self._curr_lidar_dens.pop(0)

    def evaluate_drift(self) -> Dict[str, DriftMetricResult]:
        """
        Compute Wasserstein distance and KS two-sample test against golden baseline.
        Returns detailed diagnostics across all three perception modalities.
        """
        if not self._is_calibrated or self._baseline is None:
            raise RuntimeError("Drift detector is not calibrated with a baseline distribution.")

        results = {}

        modalities = [
            ("camera_luminance", np.array(self._curr_luminance), self._baseline.luminance_sample, 255.0),
            ("depth_variance", np.array(self._curr_depth_var), self._baseline.depth_variance_sample, 10.0),
            ("lidar_density", np.array(self._curr_lidar_dens), self._baseline.lidar_density_sample, 2000.0),
        ]

        for name, current_data, baseline_data, norm_scale in modalities:
            if len(current_data) < 15:
                # Insufficient samples in current window; report nominal
                results[name] = DriftMetricResult(
                    modality=name,
                    wasserstein_dist=0.0,
                    ks_statistic=0.0,
                    p_value=1.0,
                    drift_detected=False,
                    severity=0.0,
                    recommended_action="COLLECTING_SAMPLES"
                )
                continue

            # 1. Compute 2-Wasserstein distance (normalized)
            norm_curr = current_data / norm_scale
            norm_base = baseline_data / norm_scale
            w_dist = compute_wasserstein_1d(norm_curr, norm_base)

            # 2. Compute Two-Sample Kolmogorov-Smirnov test
            ks_stat, p_val = compute_ks_2samp(norm_curr, norm_base)

            # Drift criteria: Significant KS rejection (p < alpha) AND Wasserstein distance exceeds threshold
            is_drift = (w_dist > self.wasserstein_threshold) and (p_val < self.ks_alpha)

            # Calculate continuous severity [0.0, 1.0]
            severity = min(1.0, w_dist / (self.wasserstein_threshold * 2.5))

            # Determine remediation action
            if not is_drift:
                action = "NOMINAL_OPERATION"
            elif name == "camera_luminance":
                action = "TRIGGER_OPTICAL_CLEANING_OR_EXPOSURE_RECALIBRATION"
            elif name == "depth_variance":
                action = "INSPECT_INFRARED_EMITTER_OR_SWITCH_TO_LIDAR_DEGRADED_MODE"
            else:
                action = "CLEAN_LIDAR_DOME_OR_RESTRICT_ROBOT_SPEED"

            results[name] = DriftMetricResult(
                modality=name,
                wasserstein_dist=round(w_dist, 4),
                ks_statistic=round(ks_stat, 4),
                p_value=round(p_val, 6),
                drift_detected=is_drift,
                severity=round(severity, 3),
                recommended_action=action
            )

        return results

    def is_any_drift_active(self) -> Tuple[bool, List[str]]:
        """Quick boolean gate check for control system safety."""
        report = self.evaluate_drift()
        drifting = [f"{m}: {res.recommended_action} (Wasserstein={res.wasserstein_dist})" 
                    for m, res in report.items() if res.drift_detected]
        return (len(drifting) > 0), drifting


if __name__ == "__main__":
    detector = MLOpsSensorDriftDetector(window_size=50)

    # 1. Ingest nominal data
    print("Ingesting nominal camera & LiDAR data...")
    rng = np.random.default_rng(123)
    for _ in range(50):
        detector.ingest_frame_telemetry(
            mean_luminance=float(rng.normal(128.0, 28.0)),
            depth_variance=float(rng.normal(1.5, 0.35)),
            lidar_point_count=float(rng.normal(850.0, 75.0))
        )

    has_drift, reasons = detector.is_any_drift_active()
    print(f"Nominal check -> Drift active: {has_drift}")
    assert not has_drift

    # 2. Simulate severe optical grease / lens occlusion (mean luminance drops to 25, variance collapses)
    print("\nSimulating optical lens occlusion (grease/mud)...")
    for _ in range(50):
        detector.ingest_frame_telemetry(
            mean_luminance=float(rng.normal(25.0, 5.0)),  # severe drop
            depth_variance=float(rng.normal(0.2, 0.05)),
            lidar_point_count=float(rng.normal(850.0, 75.0))
        )

    has_drift, reasons = detector.is_any_drift_active()
    print(f"Occlusion check -> Drift active: {has_drift}, Reasons: {reasons}")
    assert has_drift
    print("MLOps Sensor Drift Detector self-test passed successfully.")
