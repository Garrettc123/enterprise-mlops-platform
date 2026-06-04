"""
Auto-Retrain Check — Enterprise MLOps Platform
Garcar Enterprise | Determines if model retraining should be triggered
"""
import os
import json
import sys
from datetime import datetime, timedelta

MAX_RETRAIN_INTERVAL_DAYS = int(os.getenv("MAX_RETRAIN_INTERVAL_DAYS", "7"))
LAST_RETRAIN_FILE = "last_retrain.json"


def main():
    now = datetime.utcnow()
    if os.path.exists(LAST_RETRAIN_FILE):
        with open(LAST_RETRAIN_FILE) as f:
            data = json.load(f)
        last = datetime.fromisoformat(data["last_retrain"])
        days_since = (now - last).days
        print(f"[RETRAIN] Last retrain: {last.date()} ({days_since} days ago)")
        if days_since >= MAX_RETRAIN_INTERVAL_DAYS:
            print(f"[RETRAIN] ⚠️ Retrain overdue ({days_since}d >= {MAX_RETRAIN_INTERVAL_DAYS}d threshold)")
            print("[RETRAIN] Triggering retrain pipeline...")
            # In production: call your training orchestrator here
            sys.exit(1)
        else:
            print(f"[RETRAIN] ✅ No retrain needed yet ({MAX_RETRAIN_INTERVAL_DAYS - days_since}d remaining)")
    else:
        print("[RETRAIN] No retrain history. Saving first timestamp.")
        with open(LAST_RETRAIN_FILE, "w") as f:
            json.dump({"last_retrain": now.isoformat()}, f)


if __name__ == "__main__":
    main()
