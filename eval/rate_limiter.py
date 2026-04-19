"""eval/rate_limiter.py — Global adaptive rate limiter for API calls.

When any thread hits a 429, it calls on_rate_limit() which sets a shared
resume timestamp. All threads call wait() before each API request and block
until the pause window has elapsed.

This prevents the thundering-herd problem where per-thread backoff lets other
threads keep hammering the API while one thread is cooling down.
"""

from __future__ import annotations

import threading
import time

_DEFAULT_PAUSE = 30.0  # seconds to pause all threads on 429


class _GlobalRateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._resume_at: float = 0.0  # monotonic time when pause ends

    def on_rate_limit(self, pause_seconds: float = _DEFAULT_PAUSE) -> None:
        """Signal all threads to pause for pause_seconds.

        Safe to call from any thread. If multiple threads call this
        concurrently, the longest pause wins.
        """
        with self._lock:
            candidate = time.monotonic() + pause_seconds
            if candidate > self._resume_at:
                self._resume_at = candidate

    def wait(self) -> None:
        """Block the calling thread until the global pause has elapsed.

        Call this before every API request. Returns immediately if no
        pause is active.
        """
        while True:
            with self._lock:
                remaining = self._resume_at - time.monotonic()
            if remaining <= 0:
                return
            # Sleep in short chunks so we re-check if a longer pause arrives
            time.sleep(min(remaining, 1.0))

    def is_paused(self) -> bool:
        with self._lock:
            return time.monotonic() < self._resume_at


_limiter = _GlobalRateLimiter()

# Module-level aliases — import these directly:
#   from rate_limiter import wait, on_rate_limit
wait = _limiter.wait
on_rate_limit = _limiter.on_rate_limit
is_paused = _limiter.is_paused
