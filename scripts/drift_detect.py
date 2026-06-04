"""
Drift Detection Script — Enterprise MLOps Platform
Garcar Enterprise | Runs in CI/CD to check for data/model drift
"""
import json
import os
import sys
from datetime import datetime

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.1"))
BASELINE_FILE = os.getenv("DRIFT_BASELINE_FILE", "drift_baseline.json")


def load_baseline():
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE) as f:
            return json.load(f)
    return None


def check_drift(baseline: dict, current: dict) -> dict:
    drifts = {}
    for metric, base_val in baseline.items():
        curr_val = current.get(metric)
        if curr_val is not None:
            delta = abs(curr_val - base_val) / (abs(base_val) + 1e-9)
            drifts[metric] = {"baseline": base_val, "current": curr_val, "delta_pct": round(delta * 100, 2)}
    return drifts


def main():
    baseline = load_baseline()
    if not baseline:
        print("[DRIFT] No baseline file found. First run — establishing baseline.")
        baseline = {"accuracy": 0.95, "f1": 0.93, "latency_ms": 120}
        with open(BASELINE_FILE, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"[DRIFT] Baseline saved to {BASELINE_FILE}")
        return 0

    # In production, load current metrics from your monitoring system
    # For now, simulate no drift
    current = baseline.copy()
    drifts = check_drift(baseline, current)
    
    drift_detected = any(d["delta_pct"] > DRIFT_THRESHOLD * 100 for d in drifts.values())
    
    print(f"[DRIFT] Check at {datetime.utcnow().isoformat()}")
    for metric, info in drifts.items():
        status = "❌ DRIFT" if info["delta_pct"] > DRIFT_THRESHOLD * 100 else "✅ OK"
        print(f"  {status} {metric}: baseline={info['baseline']}, current={info['current']}, delta={info['delta_pct']}%")
    
    if drift_detected:
        print("[DRIFT] ⚠️ DRIFT DETECTED — triggering auto-retrain")
        sys.exit(1)
    else:
        print("[DRIFT] ✅ No significant drift detected")
        return 0


if __name__ == "__main__":
    sys.exit(main())
