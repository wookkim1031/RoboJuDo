import logging

import numpy as np

from robojudo.policy import policy_registry
from robojudo.policy.beyondmimic_policy import BeyondMimicPolicy
from robojudo.utils.motion_utils import _extract_yaw_quat_np

logger = logging.getLogger(__name__)


@policy_registry.register
class MjlabTrackingPolicy(BeyondMimicPolicy):
    """unitree_rl_mjlab tracking export: plain actor, one obs in, actions out."""

    _default_pose_mode: bool = False

    def set_default_pose_mode(self, enabled: bool):
        self._default_pose_mode = enabled
        logger.info(f"[MjlabTrackingPolicy] default_pose_mode={'ON' if enabled else 'OFF'}")

    def _get_command(self, env_data, ctrl_data):
        if not self._default_pose_mode:
            return super()._get_command(env_data, ctrl_data)

        default = np.asarray(self.default_dof_pos, dtype=np.float32)
        command = np.concatenate([default, np.zeros_like(default)])

        robot_anchor_quat_w = np.asarray(env_data.torso_quat, dtype=np.float32)
        anchor_quat_w = _extract_yaw_quat_np(robot_anchor_quat_w)
        anchor_pos_w = np.asarray(env_data.torso_pos, dtype=np.float32)
        robot_anchor_pos_w = anchor_pos_w.copy()

        return command, robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w, None

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ort_inputs = {self.input_names[0]: np.expand_dims(obs, axis=0).astype(np.float32)}
        ort_outputs = self.session.run([self.output_names[0]], ort_inputs)
        actions: np.ndarray = np.asarray(ort_outputs[0]).squeeze()

        actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions
        self.last_action = actions.copy()

        return actions * self.action_scales