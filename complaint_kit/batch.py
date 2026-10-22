"""Run many API calls in parallel and append each result to a JSONL file.

Shared by the routing and scam-extraction scripts. Results are written as they
arrive, so an interrupted run keeps its progress; failed calls are not written,
so rerunning the script retries exactly those.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_jsonl(path) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_jobs(jobs, work, out_path, workers: int = 4) -> list:
    """Call work(job) for each job; work returns (record or None, error or None).

    Returns the list of error messages.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    lock, errors, finished = threading.Lock(), [], 0
    with open(out_path, "a") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        for fut in as_completed(futures):
            record, error = fut.result()
            with lock:
                finished += 1
                if error:
                    errors.append(error)
                else:
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                if finished % 100 == 0 or finished == len(jobs):
                    print(f"  {finished}/{len(jobs)} done, {len(errors)} errors")
    if errors:
        print(f"\n{len(errors)} calls failed and were not saved; rerun to retry them. "
              f"First error: {errors[0]}")
    return errors
