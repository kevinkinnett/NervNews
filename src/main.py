"""Command-line entrypoint to launch the NervNews scheduler."""
from __future__ import annotations

import sys
import time

from src.scheduler.runner import run_scheduler


def main() -> int:
    scheduler = run_scheduler()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        scheduler.shutdown(wait=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
