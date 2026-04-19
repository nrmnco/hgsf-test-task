"""eval/rate_limiter.py — Global adaptive rate limiter for API calls.

When any thread hits a 429, it calls on_rate_limit() which sets a shared
resume timestamp. All threads call wait() before each API request and block
until the pause window has elapsed.

This prevents the thundering-herd problem where per-thread backoff lets other
threads keep hammering the API while one thread is cooling down.
"""

import threading
import time

_lock = threading.Lock()
_resume_at: float = 0.0  # monotonic timestamp when the global pause ends


def on_rate_limit(pause_seconds: float = 30.0) -> None:
    """Signal all threads to pause for pause_seconds.

    Safe to call from any thread. If multiple threads call this concurrently,
    the longest pause wins.
    """
    global _resume_at
    with _lock:
        candidate = time.monotonic() + pause_seconds
        if candidate > _resume_at:
            _resume_at = candidate


def wait() -> None:
    """Block the calling thread until the global pause has elapsed.

    Call this before every API request. Returns immediately if no pause is active.
    """
    while True:
        with _lock:
            remaining = _resume_at - time.monotonic()
        if remaining <= 0:
            return
        # Sleep in short chunks so we re-check if a longer pause arrives mid-sleep
        time.sleep(min(remaining, 1.0))
