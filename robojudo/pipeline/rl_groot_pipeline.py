import logging
import numpy as np

from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager, RlLocoMimicPipeline

from .groot_arm_source import GrootArmSource

logger = logging.getLogger(__name__)

@pipeline_registry.register
class RlLocoGrootPipeline(RlLocoMimicPipeline):
    def __init__(self, cfg):
        self.groot_active = False
        self.groot_source: GrootArmSource | None = None  # set via attach_groot()
 

        super().__init__(cfg=cfg)

        n = len(self.override_dof_indices)
        if n != 14:
            logger.warning("override covers %d joints, expected 14", n)

    def attach_groot(self, source: GrootArmSource) -> None:
        self.groot_source = source

    def _current_arms14(self) -> np.ndarray:
        return self.policy_manager.override_dof_pos[self.override_dof_indices].copy()

    def bail_arms_to_default(self) -> None:
        """Ramp the arms back to loco_dof_pos and stop accepting GR00T output."""
        self.groot_active = False
        if self.groot_source is not None:
            self.groot_source.stop()

        if self.policy_manager.interp_state != PolicyInterpManager.InterpState.IDLE:
            return  # a ramp is already running

        self.policy_manager._interpolate_init(
            get_target_pos=lambda: self.policy_manager.loco_dof_pos,
            durations=PolicyInterpManager.DURATIONS_MIMIC_LOCO,
            # no callback_start: the loco policy is already active
        )
        logger.warning("GR00T disengaged; arms ramping to default")

    def engage_groot(self) -> None:
        if self.groot_source is None:
            logger.error("no GrootArmSource attached")
            return
        if self.policy_manager.interp_state != PolicyInterpManager.InterpState.IDLE:
            logger.warning("interpolation in progress; not engaging")
            return
        self.groot_source.start()
        self.groot_active = True
        logger.info("GR00T engaged")
# --------------- control loop ---------------------
    def step(self, dry_run = False): 
        self.env.update()
        env_data = self.env.get_data()
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)

        commands = ctrl_data.get("COMMANDS", [])
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")

        if self.groot_source is not None:
            self.groot_source.push_state(self.env.dof_pos)

        if (
            self.groot_active
            and self.policy_manager.interp_state == PolicyInterpManager.InterpState.IDLE
        ):
            arms = self.groot_source.latest_arm_target(self._current_arms14())
            if arms is None:
                logger.error("GR00T output unavailable; bailing out")
                self.bail_arms_to_default()
            else:
                self.policy_manager.override_dof_pos[self.override_dof_indices] = arms

        if self.policy_manager.current_policy_id == self.policy_manager.policy_loco_id:
            ctrl_data["ref_dof_pos"] = self.policy.obs_adapter.fit(
                self.policy_manager.override_dof_pos
            )
            
        obs, extras = self.policy.get_observation(env_data, ctrl_data)
        pd_target = self.policy.get_pd_target(obs)

        if self.policy_manager.current_policy_id == self.policy_manager.policy_loco_id:
            pd_target[self.override_dof_indices] = self.policy_manager.override_dof_pos[
                self.override_dof_indices
            ]

        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))
 
        self.post_step_callback(env_data, ctrl_data, extras, pd_target)
 
            # -- commands -----------------------------------------------------------
 
    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        commands = ctrl_data.get("COMMANDS", [])
        for command in commands:
            if command == "[GROOT_ON]":
                self.engage_groot()
            elif command == "[GROOT_OFF]":
                self.bail_arms_to_default()
        super().post_step_callback(env_data, ctrl_data, extras, pd_target)