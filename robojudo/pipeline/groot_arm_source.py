"""
GR00T N1.7 as an upper-body override for RoboJuDo's RlLocoMimicPipeline.

Design
------
The loco policy stays active permanently and owns legs + waist.  GR00T runs in a
background thread at the dataset rate (20 Hz) and writes arm targets into
`policy_manager.override_dof_pos`, which RlLocoMimicPipeline.step() already
applies to `pd_target[override_dof_indices]` whenever the loco policy is active.

We never call switch_to_mimic().  The bail-out calls _interpolate_init()
directly, because switch_to_loco()'s guard (already-loco + IDLE) would make it a
no-op in this setup.

Joint order
-----------
GR00T's modality.json (43 dims):
    0:6 left_leg   6:12 right_leg   12:15 waist
   15:22 left_arm  22:29 left_hand  29:36 right_arm  36:43 right_hand

RoboJuDo G1 (29 dims), assumed standard Unitree order:
    0:12 legs      12:15 waist      15:22 left_arm   22:29 right_arm

=> arm override vector (14) = concat(groot[15:22], groot[29:36])
=> override_dof_indices = range(-14, 0) = 15:29  with upper_dof_num=14

VERIFY THIS against the joint names in your G1 config before running on
hardware.  A silent permutation produces plausible-looking wrong motion.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# --- dataset facts (g1-pick-apple) ---------------------------------------
GROOT_FPS = 20
GROOT_DT = 1.0 / GROOT_FPS

G_LEFT_ARM = slice(15, 22)
G_RIGHT_ARM = slice(29, 36)
G_LEFT_LEG = slice(0, 6)
G_RIGHT_LEG = slice(6, 12)
G_WAIST = slice(12, 15)

# RoboJuDo 29-dof indices
RJ_LEGS = slice(0, 12)
RJ_WAIST = slice(12, 15)
RJ_LEFT_ARM = slice(15, 22)
RJ_RIGHT_ARM = slice(22, 29)


LEFT_HAND_FILL = np.array([-0.151, -0.316, -0.146, -0.325, 0.016, 0.147, 0.161], dtype=np.float32)
RIGHT_HAND_FILL = np.array([0.018, 0.031, 0.030, 0.040, 0.125, -0.030, -0.025], dtype=np.float32)

def rj29_to_groot43(dof29: np.ndarray) -> np.ndarray:
    """Build GR00T's 43-dim state vector from RoboJuDo's 29-dim dof_pos.

    Hand blocks are zero-filled: the dataset used Unitree tri-finger hands,
    which is not the hardware here.
    """
    g = np.zeros(43, dtype=np.float32)
    g[G_LEFT_LEG] = dof29[0:6]
    g[G_RIGHT_LEG] = dof29[6:12]
    g[G_WAIST] = dof29[RJ_WAIST]
    g[G_LEFT_ARM] = dof29[RJ_LEFT_ARM]
    g[G_RIGHT_ARM] = dof29[RJ_RIGHT_ARM]
    g[22:29] = LEFT_HAND_FILL
    g[36:43] = RIGHT_HAND_FILL
    return g


def groot43_to_arms14(action43: np.ndarray) -> np.ndarray:
    """Extract the 14 arm joints in RoboJuDo override order (left then right)."""
    return np.concatenate([action43[G_LEFT_ARM], action43[G_RIGHT_ARM]])


class GrootArmSource:
    """Runs GR00T in a background thread and serves interpolated arm targets.

    The pipeline pushes state/frames in (push_state / the camera callback) and
    pulls targets out (latest_arm_target).  Inference never blocks the control
    loop.
    """

    def __init__(
        self,
        policy,
        camera_read,
        prompt: str,
        execution_horizon: int = 16,
        stale_after: float = 0.5,
        max_step_per_tick: float = 0.02,
    ):
        """
        policy           : PolicyClient or Gr00tPolicy, must expose get_action(obs)
        camera_read      : callable -> uint8 HxWx3 RGB frame matching the dataset view
        prompt           : language instruction
        execution_horizon: chunk steps consumed before re-planning
        stale_after      : seconds without a fresh chunk before going inactive
        max_step_per_tick: rad, per-control-tick rate clamp on the override
        """
        self._policy = policy
        self._camera_read = camera_read
        self._prompt = prompt
        self._horizon = execution_horizon
        self._stale_after = stale_after
        self._max_step = max_step_per_tick

        self._lock = threading.Lock()
        self._state29 = None          # latest dof_pos pushed by the pipeline
        self._chunk = None            # (horizon, 14) arm targets
        self._chunk_t0 = None         # monotonic time chunk step 0 is valid at
        self._last_output = None      # for rate clamping
        self._thread = None
        self._running = False
        self._infer_latency = None

    # -- pipeline-facing API ------------------------------------------------

    def push_state(self, dof_pos29: np.ndarray) -> None:
        with self._lock:
            self._state29 = np.asarray(dof_pos29, dtype=np.float32).copy()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("GrootArmSource started")

    def stop(self) -> None:
        self._running = False
        with self._lock:
            self._chunk = None
            self._chunk_t0 = None
            self._last_output = None
        logger.info("GrootArmSource stopped")

    def latest_arm_target(self, current_arms14: np.ndarray) -> np.ndarray | None:
        """Interpolated 14-dim arm target for this control tick, or None.

        None means: not running, no chunk yet, or the chunk has gone stale.
        The caller must treat None as 'do not write the override'.
        """
        if not self._running:
            return None

        with self._lock:
            chunk = self._chunk
            t0 = self._chunk_t0

        if chunk is None or t0 is None:
            return None

        now = time.monotonic()
        age = now - t0
        if age > self._stale_after + self._horizon * GROOT_DT:
            logger.warning("GR00T chunk stale (%.2fs) - override withheld", age)
            return None

        # Position inside the 20 Hz chunk, linearly interpolated to the caller's rate.
        pos = age / GROOT_DT
        i = int(np.floor(pos))
        if i >= len(chunk) - 1:
            target = chunk[-1]
        else:
            alpha = pos - i
            target = (1.0 - alpha) * chunk[i] + alpha * chunk[i + 1]

        # Rate clamp against what we actually commanded last tick.
        ref = self._last_output if self._last_output is not None else current_arms14
        delta = np.clip(target - ref, -self._max_step, self._max_step)
        out = ref + delta
        self._last_output = out
        return out

    @property
    def infer_latency(self) -> float | None:
        return self._infer_latency

    # -- background inference ----------------------------------------------

    def _build_observation(self, state43: np.ndarray, frame: np.ndarray) -> dict:
        obs = {
            "state.left_leg": state43[G_LEFT_LEG][None, :],
            "state.right_leg": state43[G_RIGHT_LEG][None, :],
            "state.waist": state43[G_WAIST][None, :],
            "state.left_arm": state43[G_LEFT_ARM][None, :],
            "state.left_hand": state43[22:29][None, :],
            "state.right_arm": state43[G_RIGHT_ARM][None, :],
            "state.right_hand": state43[36:43][None, :],
            "video.rs_view": frame[None, ...],
            "annotation.human.task_description": [self._prompt],
        }
        return obs

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                state29 = None if self._state29 is None else self._state29.copy()

            if state29 is None:
                time.sleep(0.01)
                continue

            try:
                frame = self._camera_read()
                state43 = rj29_to_groot43(state29)
                obs = self._build_observation(state43, frame)

                t_start = time.monotonic()
                action, _ = self._policy.get_action(obs)
                self._infer_latency = time.monotonic() - t_start

                left = np.asarray(action["action.left_arm"])
                right = np.asarray(action["action.right_arm"])
                if left.ndim == 3:            # (B, T, D) -> unbatch
                    left, right = left[0], right[0]
                chunk = np.concatenate(
                    [left[: self._horizon], right[: self._horizon]], axis=-1
                ).astype(np.float32)

                if not np.all(np.isfinite(chunk)):
                    logger.error("non-finite GR00T output; dropping chunk")
                    continue

                with self._lock:
                    # Anchor at t_start: the chunk is relative to the state we
                    # sent in, not to the moment inference finished.
                    self._chunk = chunk
                    self._chunk_t0 = t_start

            except Exception:
                logger.exception("GR00T inference failed; override will go stale")
                time.sleep(0.1)
                continue

            # Re-plan just before the current chunk runs out.
            sleep = max(0.0, (self._horizon - 1) * GROOT_DT - (time.monotonic() - t_start))
            time.sleep(sleep)