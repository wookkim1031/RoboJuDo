
# plus RlPipelineCfg and cfg_registry, copy the import lines from g1_cfg.py

from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import BeyondMimicCtrlCfg, KeyboardCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.tools.tool_cfgs import DoFConfig

from .ctrl.g1_beyondmimic_ctrl_cfg import G1BeyondmimicCtrlCfg
from .env.g1_mujuco_env_cfg import G1MujocoEnvCfg
from .env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg  # noqa: 
from .policy.g1_beyondmimic_policy_cfg import G1BeyondMimicPolicyCfg
from robojudo.pipeline.pipeline_cfgs import (
    RlLocoMimicPipelineCfg,  # noqa: F401
    RlMultiPolicyPipelineCfg,  # noqa: F401
    RlPipelineCfg,  # noqa: F401
)
from .policy.g1_unitree_policy_cfg import G1UnitreePolicyCfg, G1UnitreeWoGaitPolicyCfg  # noqa: F401
 

class G1MjlabDoF(DoFConfig):
    """SDK joint order: left leg 6, right leg 6, waist 3, left arm 7, right arm 7."""

    joint_names: list[str] = [
        *["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
          "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],
        *["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
          "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
        *["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        *["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
          "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
          "left_wrist_yaw_joint"],
        *["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
          "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
          "right_wrist_yaw_joint"],
    ]

    default_pos: list[float] | None = [
        *[-0.312, 0.0, 0.0, 0.669, -0.363, 0.0],
        *[-0.312, 0.0, 0.0, 0.669, -0.363, 0.0],
        *[0.0, 0.0, 0.0],
        *[0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0],
        *[0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0],
    ]

    stiffness: list[float] | None = [
        *[40.179, 99.098, 40.179, 99.098, 28.501, 28.501],
        *[40.179, 99.098, 40.179, 99.098, 28.501, 28.501],
        *[40.179, 28.501, 28.501],
        *[14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778],
        *[14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778],
    ]

    damping: list[float] | None = [
        *[2.558, 6.309, 2.558, 6.309, 1.814, 1.814],
        *[2.558, 6.309, 2.558, 6.309, 1.814, 1.814],
        *[2.558, 1.814, 1.814],
        *[0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068],
        *[0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068],
    ]
    
class G1MjlabPolicyCfg(G1BeyondMimicPolicyCfg):
    policy_type: str = "MjlabTrackingPolicy"
    policy_name: str = "mjlab_walk"
    obs_dof: DoFConfig = G1MjlabDoF()
    action_dof: DoFConfig = obs_dof
    action_beta: float = 1.0
    without_state_estimator: bool = True
    use_modelmeta_config: bool = False
    use_motion_from_model: bool = False

    action_scales: list[float] = [
        # left leg: hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll
        *[0.5475464629911068, 0.35066146637882434, 0.5475464629911068,
          0.35066146637882434, 0.43857731392336724, 0.43857731392336724],
        # right leg
        *[0.5475464629911068, 0.35066146637882434, 0.5475464629911068,
          0.35066146637882434, 0.43857731392336724, 0.43857731392336724],
        # waist: yaw, roll, pitch
        *[0.5475464629911068, 0.43857731392336724, 0.43857731392336724],
        # left arm: shoulder p/r/y, elbow, wrist roll, wrist pitch, wrist yaw
        *[0.43857731392336724, 0.43857731392336724, 0.43857731392336724,
          0.43857731392336724, 0.43857731392336724,
          0.07450087032950714, 0.07450087032950714],
        # right arm
        *[0.43857731392336724, 0.43857731392336724, 0.43857731392336724,
          0.43857731392336724, 0.43857731392336724,
          0.07450087032950714, 0.07450087032950714],
    ]

class G1MjlabCtrlCfg(G1BeyondmimicCtrlCfg):
    motion_name: str = "walk_chunk_0000"
    override_robot_anchor_pos: bool = True

    motion_cfg: BeyondMimicCtrlCfg.MotionCommandCfg = BeyondMimicCtrlCfg.MotionCommandCfg(
        anchor_body_name="torso_link",
        body_names=[
            "pelvis",
            "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
            "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
            "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
        ],
        body_names_all=[
            "pelvis",
            "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
            "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
            "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
            "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
            "waist_yaw_link", "waist_roll_link", "torso_link",
            "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
            "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
            "left_wrist_yaw_link",
            "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
            "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
            "right_wrist_yaw_link",
        ],
    )

@cfg_registry.register
class g1_mjlab_walk(RlPipelineCfg):
    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(visualize_extras=False)
    ctrl: list[KeyboardCtrlCfg | G1MjlabCtrlCfg] = [KeyboardCtrlCfg(), G1MjlabCtrlCfg()]
    policy: G1MjlabPolicyCfg = G1MjlabPolicyCfg()

@cfg_registry.register
class g1_mjlab_locomimic(RlLocoMimicPipelineCfg):
    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(visualize_extras=False)
    ctrl: list[KeyboardCtrlCfg | G1MjlabCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={"]": "[POLICY_LOCO]", "[": "[POLICY_MIMIC]"}
        ),
        G1MjlabCtrlCfg(),
    ]
    loco_policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()
    mimic_policies: list[G1MjlabPolicyCfg] = [G1MjlabPolicyCfg()]

@cfg_registry.register
class g1_mjlab_locomimic_real(RlLocoMimicPipelineCfg):
    robot: str = "g1"
    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeEnv",
        unitree=G1UnitreeCfg(net_if="eth0"),   # NEED TO SET THIS!
    )
    ctrl: list[KeyboardCtrlCfg | G1MjlabCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers={"i": "[SIM_REBORN]", "o": "[SHUTDOWN]",
                      "]": "[POLICY_LOCO]", "[": "[POLICY_MIMIC]"}
        ),
        G1MjlabCtrlCfg(),
    ]
    loco_policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()
    mimic_policies: list[G1MjlabPolicyCfg] = [G1MjlabPolicyCfg()]