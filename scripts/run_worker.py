"""Run MANTRA background worker loop."""

import signal
import time

from mantra.interfaces.api import api_server


def _stop(signum, frame):
    api_server._worker.stop()
    raise SystemExit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    api_server._worker.start()

    try:
        while True:
            time.sleep(1.0)
    finally:
        api_server._worker.stop()
