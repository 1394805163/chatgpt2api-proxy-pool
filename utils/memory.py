"""Small process-memory cleanup helpers used after image work."""

from __future__ import annotations

import gc


def release_process_memory() -> None:
    """Release unreachable Python objects and ask glibc to trim free arenas.

    The call is deliberately best-effort: ``malloc_trim`` is available in the
    Linux/glibc runtime used by the container, but is not present on every
    development platform.  Image responses call this only after their result
    has been reduced to URLs, so it does not alter response contents.
    """
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
