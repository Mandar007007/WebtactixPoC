"""Force UTF-8 stdout/stderr so Jira icon chars do not crash on Windows."""
from __future__ import annotations

import sys


def configure_utf8_stdio() -> None:
    """Reconfigure stdio to UTF-8 with line buffering and error replacement."""
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        reconfigure = getattr(stream, 'reconfigure', None)
        if callable(reconfigure):
            try:
                reconfigure(
                    encoding='utf-8',
                    errors='replace',
                    line_buffering=True,
                )
            except Exception:
                pass
