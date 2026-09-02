"""Headless replacement for RoboJuDo's pynput-based KeyboardThread.

The original imports ``pynput``, which needs an X connection at import time and
therefore cannot run on a headless pod.  This version imports nothing exotic and
instead injects *scheduled* synthetic key-release events into the same queue,
which is all ``KeyboardCtrl.process_triggers`` looks at (it fires on
``pressed == False`` and matches ``event["name"]`` against ``cfg.triggers``).

Schedule is read from the environment so no code change is needed per run::

    ROBOJUDO_AUTOKEYS="3.0:r,10.0:o"

means: 3.0 s after start send "r", 10.0 s after start send "o".  With the
``g1_protomotions_tracker`` triggers that is ``[MOTION_RESET]`` (start the
motion) followed by ``[SHUTDOWN]`` (which finalises the recorded mp4).

Set ROBOJUDO_AUTOKEYS="" to disable and just idle.
"""

import logging
import os
import time
from queue import Queue
from threading import Thread

logger = logging.getLogger(__name__)

DEFAULT_SCHEDULE = "3.0:r,10.0:o"


def _parse_schedule(spec: str) -> list[tuple[float, str]]:
    """Parse "3.0:r,10.0:o" into [(3.0, "r"), (10.0, "o")], sorted by time."""
    out: list[tuple[float, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            delay_str, key = item.split(":", 1)
            out.append((float(delay_str), key))
        except ValueError:
            logger.warning("[KeyboardThread] ignoring malformed schedule item: %r", item)
    return sorted(out, key=lambda p: p[0])


class KeyboardThread(Thread):
    """Drop-in stand-in for the pynput version.

    Same constructor signature and same event dict shape, so ``KeyboardCtrl``
    needs no modification.
    """

    def __init__(self, event_queue: Queue):
        super().__init__(name="KeyboardThread", daemon=True)
        self.event_queue = event_queue
        self.schedule = _parse_schedule(os.environ.get("ROBOJUDO_AUTOKEYS", DEFAULT_SCHEDULE))
        if self.schedule:
            logger.info("[KeyboardThread] headless mode, scheduled keys: %s", self.schedule)
        else:
            logger.info("[KeyboardThread] headless mode, no scheduled keys")

    def _emit(self, key_name: str):
        # KeyboardCtrl triggers on key *release*, so send press then release.
        now = time.time()
        self.event_queue.put(
            {"type": "keyboard", "name": key_name, "pressed": True, "timestamp": now}
        )
        self.event_queue.put(
            {"type": "keyboard", "name": key_name, "pressed": False, "timestamp": now}
        )
        logger.info("[KeyboardThread] injected key %r", key_name)

    def run(self):
        t0 = time.time()
        for delay, key_name in self.schedule:
            remaining = delay - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)
            self._emit(key_name)

        # Stay alive so the thread behaves like the original listener.
        while True:
            time.sleep(1.0)

    def get_key_name(self, key):  # kept for API compatibility
        try:
            return key.char if key.char is not None else str(key)
        except AttributeError:
            return str(key)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    q: Queue = Queue()
    KeyboardThread(q).start()
    print("Waiting for scheduled events...")
    while True:
        print(q.get())
