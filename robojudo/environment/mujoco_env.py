import logging
import time

import mujoco
# import mujoco_viewer
import numpy as np

from pathlib import Path
from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)
import logging
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import mujoco.viewer as mujoco_viewer
from mujoco_viewer import MujocoViewer as StandaloneMujocoViewer
import numpy as np


logger = logging.getLogger(__name__)

import logging
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import mujoco.viewer as mujoco_viewer
import numpy as np


logger = logging.getLogger(__name__)


class _MarkerMixin:
    """Shared marker handling for LiveViewer and OffscreenViewer."""

    def _init_marker_state(self):
        self._markers = []

    def add_marker(self, *args, **kwargs):
        """
        Compatible with mujoco_viewer.MujocoViewer.add_marker().

        Supports both:

            viewer.add_marker(
                pos=...,
                size=...,
                mat=...,
                rgba=...,
                type=...,
            )

        and positional arguments in this order:

            pos, size, mat, rgba, type, label
        """

        positional_names = (
            "pos",
            "size",
            "mat",
            "rgba",
            "type",
            "label",
        )

        for name, value in zip(positional_names, args):
            kwargs.setdefault(name, value)

        if self.is_alive:
            self._markers.append(dict(kwargs))

    @staticmethod
    def _as_vec3(value, default):
        if value is None:
            value = default

        array = np.asarray(value, dtype=np.float64).reshape(-1)

        if array.size == 1:
            array = np.repeat(array, 3)

        if array.size != 3:
            raise ValueError(
                f"Expected a 3-element vector, got {array}"
            )

        return array

    @staticmethod
    def _as_mat3(value):
        if value is None:
            return np.eye(3, dtype=np.float64).reshape(-1)

        array = np.asarray(value, dtype=np.float64)

        if array.shape == (3, 3):
            return array.reshape(-1)

        if array.size == 9:
            return array.reshape(-1)

        raise ValueError(
            f"Expected a 3x3 matrix, got shape {array.shape}"
        )

    @staticmethod
    def _as_rgba(value):
        if value is None:
            value = [1.0, 0.0, 0.0, 1.0]

        rgba = np.asarray(value, dtype=np.float32).reshape(-1)

        if rgba.size != 4:
            raise ValueError(
                f"Marker rgba must contain 4 values, got {rgba}"
            )

        return rgba

    @staticmethod
    def _get_geom_type(value):
        if value is None:
            return mujoco.mjtGeom.mjGEOM_SPHERE

        if not isinstance(value, str):
            return value

        geom_name = value.upper()

        if not geom_name.startswith("MJGEOM_"):
            geom_name = f"MJGEOM_{geom_name}"

        try:
            return getattr(mujoco.mjtGeom, geom_name)
        except AttributeError as exc:
            raise ValueError(
                f"Unknown MuJoCo geometry type: {value}"
            ) from exc

    @staticmethod
    def _init_geom(geom, marker):
        geom_type = _MarkerMixin._get_geom_type(
            marker.get("type")
        )

        size = _MarkerMixin._as_vec3(
            marker.get("size"),
            default=[0.05, 0.05, 0.05],
        )

        pos = _MarkerMixin._as_vec3(
            marker.get("pos"),
            default=[0.0, 0.0, 0.0],
        )

        mat = _MarkerMixin._as_mat3(
            marker.get("mat")
        )

        rgba = _MarkerMixin._as_rgba(
            marker.get("rgba")
        )

        mujoco.mjv_initGeom(
            geom,
            geom_type,
            size,
            pos,
            mat,
            rgba,
        )

        label = marker.get("label")
        if label is not None and hasattr(geom, "label"):
            geom.label = str(label)


class OffscreenViewer(_MarkerMixin):
    """
    Headless MuJoCo viewer.

    Renders with mujoco.Renderer and writes frames to an MP4 file.
    """

    def __init__(
        self,
        model,
        data,
        path,
        w=1200,
        h=900,
        fps=50,
    ):
        # MuJoCo's renderer cannot use a buffer smaller than the requested
        # rendering resolution.
        model.vis.global_.offwidth = max(
            model.vis.global_.offwidth,
            w,
        )
        model.vis.global_.offheight = max(
            model.vis.global_.offheight,
            h,
        )

        self.model = model
        self.data = data
        self._width = w
        self._height = h
        self._fps = fps

        self.renderer = mujoco.Renderer(
            model,
            height=h,
            width=w,
        )

        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)

        self.is_alive = True
        self._paused = False

        self._path = Path(path)

        # If a directory is supplied, write a default video filename.
        if self._path.suffix == "":
            self._path = self._path / "rollout.mp4"

        # Fixes FileNotFoundError when ./tmp does not exist.
        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._writer = imageio.get_writer(
            str(self._path),
            fps=fps,
        )

        self._init_marker_state()

        logger.info(
            "OffscreenViewer recording to %s (%dx%d @ %d fps)",
            self._path,
            w,
            h,
            fps,
        )

    def _add_markers_to_scene(self):
        scene = self.renderer.scene

        for marker in self._markers:
            if scene.ngeom >= scene.maxgeom:
                logger.warning(
                    "Offscreen render scene is full; skipping marker"
                )
                break

            geom = scene.geoms[scene.ngeom]

            self._init_geom(
                geom,
                marker,
            )

            scene.ngeom += 1

    def render(self):
        if not self.is_alive:
            return

        self.renderer.update_scene(
            self.data,
            camera=self.cam,
        )

        # Markers must be added after update_scene().
        self._add_markers_to_scene()

        frame = self.renderer.render()
        self._writer.append_data(frame)

        # Markers apply only to the current frame.
        self._markers.clear()

    def close(self):
        if not self.is_alive:
            return

        self.is_alive = False

        try:
            self._writer.close()
        finally:
            if hasattr(self.renderer, "close"):
                self.renderer.close()

        logger.info(
            "OffscreenViewer wrote %s",
            self._path,
        )


class LiveViewer(_MarkerMixin):
    """
    Interactive MuJoCo viewer.

    This class has the same public interface as OffscreenViewer:

        viewer.cam
        viewer.is_alive
        viewer._paused
        viewer.add_marker()
        viewer.render()
        viewer.close()
    """

    def __init__(
        self,
        model,
        data,
        path=None,
        w=1200,
        h=900,
        fps=50,
    ):
        # These arguments are accepted for compatibility with
        # OffscreenViewer but are not needed by launch_passive().
        del path, w, h, fps

        self.model = model
        self.data = data

        self._alive = True
        self._paused = False

        self._init_marker_state()

        # Important:
        # import mujoco.viewer as mujoco_viewer
        #
        # is required because import mujoco alone does not always expose
        # the viewer module as mujoco.viewer.
        self._viewer = mujoco_viewer.launch_passive(
            model,
            data,
        )

        # MujocoEnv already uses:
        #
        # self.viewer.cam.distance
        # self.viewer.cam.elevation
        # self.viewer.cam.azimuth
        # self.viewer.cam.lookat
        #
        # Therefore expose the native viewer camera directly.
        self.cam = self._viewer.cam

        logger.info(
            "Live MuJoCo viewer started"
        )

    @property
    def is_alive(self):
        if not self._alive:
            return False

        try:
            return bool(
                self._viewer.is_running()
            )
        except Exception:
            return False

    def _add_markers_to_scene(self):
        scene = self._viewer.user_scn

        # Clear markers from the previous frame.
        scene.ngeom = 0

        for marker in self._markers:
            if scene.ngeom >= scene.maxgeom:
                logger.warning(
                    "Live viewer user scene is full; skipping marker"
                )
                break

            geom = scene.geoms[scene.ngeom]

            self._init_geom(
                geom,
                marker,
            )

            scene.ngeom += 1

    def render(self):
        if not self.is_alive:
            return

        # user_scn must be modified while holding the viewer lock.
        with self._viewer.lock():
            self._add_markers_to_scene()

        # Synchronize model/data/camera/user markers with the GLFW window.
        self._viewer.sync()

        # Markers apply only to the current frame.
        self._markers.clear()

    def close(self):
        if not self._alive:
            return

        self._alive = False

        try:
            self._viewer.close()
        except Exception:
            logger.exception(
                "Failed to close Live MuJoCo viewer"
            )

        logger.info(
            "Live MuJoCo viewer closed"
        )


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        self.model = mujoco.MjModel.from_xml_path(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        # # self.viewer = mujoco_viewer.MujocoViewer(
        # #     self.model,
        # #     self.data,
        # #     width=1200,
        # #     height=900,
        # #     hide_menus=True,
        # # )
        # # self.viewer.cam.distance = 3.0
        # # self.viewer.cam.elevation = -10.0
        # # self.viewer.cam.azimuth = 180.0
        # # self.viewer._paused = True
        # self.viewer = LiveViewer(
        #     self.model,
        #     self.data,
        #     path=Path("./tmp") / "rollout.mp4",
        #     w=1200,
        #     h=900,
        #     fps=round(1.0 / self.control_dt),
        # )
        import os
        viewer_type = cfg_env.viewer.lower()
        print("Viewer_type", viewer_type)
        if viewer_type == "live":
            self.viewer = LiveViewer(
                self.model,
                self.data,
                path=Path("./tmp") / "rollout.mp4",
                w=1200,
                h=900,
                fps=round(1.0 / self.control_dt),
            )

        elif viewer_type == "offscreen":
            self.viewer = OffscreenViewer(
                self.model,
                self.data,
                path=Path("./tmp") / "rollout.mp4",
                w=1200,
                h=900,
                fps=round(1.0 / self.control_dt),
            )

        elif viewer_type == "window":
            self.viewer = StandaloneMujocoViewer(self.model, self.data,
                                                    width=1200, height=900, hide_menus=True)
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -10.0
            self.viewer.cam.azimuth = 180.0
            
        elif os.environ.get("DISPLAY"):
            self.viewer = LiveViewer(
                self.model,
                self.data,
                path=Path("./tmp") / "rollout.mp4",
                w=1200,
                h=900,
                fps=round(1.0 / self.control_dt),
            )

        else:
            logger.warning(
                "DISPLAY is not set; using OffscreenViewer instead of LiveViewer"
            )

            self.viewer = OffscreenViewer(
                self.model,
                self.data,
                path=Path("./tmp") / "rollout.mp4",
                w=1200,
                h=900,
                fps=round(1.0 / self.control_dt),
            )

        if cfg_env.visualize_extras:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self.last_time = time.time()
        self.random_heading = cfg_env.random_heading

        self._apply_random_heading()

        self.update()  # get initial state

    def _apply_random_heading(self):
        """Rotate the root body by a random yaw if random_heading is enabled."""
        if not self.random_heading:
            return
        yaw = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        q = self.data.qpos[3:7].copy()  # MuJoCo [w, x, y, z]
        # Pre-multiply by yaw rotation q_yaw=[c,0,0,s]: q_new = q_yaw ⊗ q
        self.data.qpos[3] = c * q[0] - s * q[3]
        self.data.qpos[4] = c * q[1] - s * q[2]
        self.data.qpos[5] = c * q[2] + s * q[1]
        self.data.qpos[6] = c * q[3] + s * q[0]

    def reborn(self, init_qpos=None):
        if init_qpos is not None:
            self.data.qpos[0:7] = init_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
            self._apply_random_heading()
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def self_check(self):
        pass

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs :]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs :]

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        ang_vel = self.data.qvel.astype(np.float32)[3:6]
        base_pos = self.data.qpos.astype(np.float32)[:3]
        lin_vel = self.data.qvel.astype(np.float32)[0:3]

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)

        self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        if self.viewer.is_alive:
            self.viewer.render()

        for _ in range(self.sim_decimation):
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)

            self.data.ctrl = torque

            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.update(simple=True)
        self.update(simple=False)

    def shutdown(self):
        self.viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)