"""
View HF Jobs logs on Windows without Unicode encoding errors.

Usage:
    python view_logs.py <job_id>
    python view_logs.py <job_id> --tail 50
    python view_logs.py <job_id> --no-progress
    python view_logs.py <job_id> --timeout 30
"""

import sys
import argparse
import threading

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="View HF Jobs logs (Windows-safe)")
    parser.add_argument("job_id", help="HF Job ID")
    parser.add_argument(
        "--tail", type=int, default=0,
        help="Only show last N lines (0 = show all)",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Filter out progress bar lines",
    )
    parser.add_argument(
        "--timeout", type=int, default=15,
        help="Seconds to collect logs before printing (default: 15)",
    )
    args = parser.parse_args()

    api = HfApi()

    # First check job status
    jobs = api.list_jobs()
    job = next((j for j in jobs if j.id == args.job_id), None)
    if job:
        print(f"Job: {job.id}")
        print(f"Status: {job.status.stage}")
        print(f"Flavor: {job.flavor}")
        print(f"URL: https://huggingface.co/jobs/{job.owner.name}/{job.id}")
        print("-" * 60)

    # Collect logs with a timeout (fetch_job_logs streams indefinitely)
    collected = []

    def collect():
        try:
            for log in api.fetch_job_logs(job_id=args.job_id):
                collected.append(log)
        except Exception as e:
            collected.append(f"[ERROR] {e}")

    t = threading.Thread(target=collect, daemon=True)
    t.start()
    t.join(timeout=args.timeout)

    if not collected:
        print("No logs available yet. Job may still be starting.")
        return

    lines = collected

    # Filter progress bars if requested
    if args.no_progress:
        lines = [
            l for l in lines
            if "[A" not in l
            and not ("|" in l and ("%" in l or "it/s" in l))
        ]

    # Tail
    if args.tail > 0:
        lines = lines[-args.tail:]

    for line in lines:
        print(line)

    print(f"\n--- Collected {len(collected)} total log lines in {args.timeout}s ---")


if __name__ == "__main__":
    main()
