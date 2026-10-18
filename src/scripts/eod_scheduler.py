"""
Automated EOD Daemon & Cron Scheduler for Docker.
==================================================
Runs inside the `cron` container service.
Triggers the full EOD pipeline every weekday (Monday-Friday) at 18:00 IST (NSE market close settlement).

Usage:
  python scripts/eod_scheduler.py          # Starts daemon loop (default)
  python scripts/eod_scheduler.py --now    # Executes immediate EOD run then exits
"""

import os
import sys
import time
import signal
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR) if os.path.exists(os.path.join(os.path.dirname(SRC_DIR), 'data')) else SRC_DIR
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.eod_pipeline import run_eod_pipeline

RUNNING = True

def handle_shutdown(signum, frame):
    global RUNNING
    print(f"\n[Scheduler] Caught signal {signum}. Shutting down gracefully...")
    RUNNING = False

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)


def get_next_run_time(target_time_str: str = "18:00", tz_name: str = "Asia/Kolkata") -> datetime:
    """Calculates the next weekday execution datetime in Asia/Kolkata timezone."""
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    hour, minute = map(int, target_time_str.split(":"))

    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)

    # Advance over weekends (Saturday=5, Sunday=6) to Monday
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)

    return candidate


def run_scheduler_daemon():
    schedule_time_str = os.getenv("EOD_SCHEDULE_TIME", "18:00")
    tz_name = os.getenv("TZ", "Asia/Kolkata")

    print("=" * 70)
    print("  INDIA QUANT REGIME DETECTOR — EOD DOCKER SCHEDULER")
    print(f"  Schedule: Every Weekday (Mon–Fri) at {schedule_time_str} {tz_name}")
    print("=" * 70)

    while RUNNING:
        next_run = get_next_run_time(schedule_time_str, tz_name)
        now = datetime.now(ZoneInfo(tz_name))
        wait_seconds = (next_run - now).total_seconds()
        hours, remainder = divmod(int(wait_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\n[Scheduler] Current time : {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"[Scheduler] Next EOD run : {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')} (in {hours}h {minutes}m {seconds}s)")

        # Sleep in small slices so container SIGTERM is handled immediately
        while wait_seconds > 0 and RUNNING:
            sleep_chunk = min(wait_seconds, 10.0)
            time.sleep(sleep_chunk)
            now = datetime.now(ZoneInfo(tz_name))
            wait_seconds = (next_run - now).total_seconds()

        if not RUNNING:
            break

        print(f"\n[Scheduler] Triggering scheduled EOD run at {now.strftime('%Y-%m-%d %H:%M:%S %Z')}...")
        try:
            run_eod_pipeline()
        except Exception as e:
            print(f"[Scheduler] Error during EOD pipeline: {e}")

        # Sleep a minute to avoid double-triggering in the same minute
        time.sleep(65)

    print("[Scheduler] Stopped cleanly.")


if __name__ == "__main__":
    if "--now" in sys.argv:
        print("[Scheduler] Running immediate one-shot EOD execution...")
        run_eod_pipeline()
    else:
        run_scheduler_daemon()
