"""Bounded damped least-squares IK for absolute base-to-link7 xyzw poses."""
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from inference.mujoco_visualization import MujocoKinematicFK


class PoseIK:
    def __init__(self, config, hardware):
        if config['ee_body_name'] != 'link7' or config.get('ee_site_name'):
            raise ValueError('pure pi0 IK requires the link7 body frame (no tool site offset)')
        self.fk = MujocoKinematicFK(SimpleNamespace(**config))
        mj, model = self.fk.mujoco, self.fk.model
        self.base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'base_link')
        if self.base_id < 0:
            raise ValueError('pure pi0 IK model requires base_link')
        ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name) for name in config['robot_joint_names']]
        if len(ids) != 7:
            raise ValueError('pure pi0 IK requires seven robot joints')
        self.dofs = model.jnt_dofadr[ids]
        self.low = np.maximum(hardware['q_min'], model.jnt_range[ids, 0])
        self.high = np.minimum(hardware['q_max'], model.jnt_range[ids, 1])
        if np.any(self.low >= self.high):
            raise ValueError('IK joint bounds have no valid intersection')

    def pose(self, q):
        fk = self.fk
        fk.scratch_data.qpos[fk.addresses] = q
        fk.mujoco.mj_forward(fk.model, fk.scratch_data)
        base_rotation = fk.scratch_data.xmat[self.base_id].reshape(3, 3)
        rotation = base_rotation.T @ fk.scratch_data.xmat[fk.ee_id].reshape(3, 3)
        quat = Rotation.from_matrix(rotation).as_quat()
        if quat[3] < 0:
            quat *= -1
        position = base_rotation.T @ (fk.scratch_data.xpos[fk.ee_id] - fk.scratch_data.xpos[self.base_id])
        return np.r_[position, quat]

    def solve(self, target, seed):
        target, q = np.asarray(target, dtype=float), np.asarray(seed, dtype=float).copy()
        if target.shape != (7,) or q.shape != (7,) or not np.isfinite(target).all() or not np.isfinite(q).all():
            raise ValueError('IK target and seed must be finite seven-vectors')
        if abs(np.linalg.norm(target[3:]) - 1) > .15:
            raise ValueError('IK target must contain an xyzw unit quaternion')
        q = np.clip(q, self.low, self.high)
        desired = Rotation.from_quat(target[3:]).as_matrix()
        fk = self.fk
        jp, jr = np.zeros((3, fk.model.nv)), np.zeros((3, fk.model.nv))
        for _ in range(101):
            pose = self.pose(q)
            dp = target[:3] - pose[:3]
            dr = Rotation.from_matrix(desired @ Rotation.from_quat(pose[3:]).as_matrix().T).as_rotvec()
            if np.linalg.norm(dp) <= 1e-4 and np.linalg.norm(dr) <= 1e-3:
                return q
            fk.mujoco.mj_jacBody(fk.model, fk.scratch_data, jp, jr, fk.ee_id)
            base_rotation = fk.scratch_data.xmat[self.base_id].reshape(3, 3)
            jac = np.vstack((base_rotation.T @ jp[:, self.dofs], base_rotation.T @ jr[:, self.dofs]))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-6, np.r_[dp, dr])
            delta *= min(1., .2 / max(np.linalg.norm(delta), 1e-12))
            q = np.clip(q + delta, self.low, self.high)
        raise RuntimeError(f'pi0 IK did not converge: position_error={np.linalg.norm(dp):.6f}m '
                           f'rotation_error={np.linalg.norm(dr):.6f}rad')

    def chunk(self, poses, seed):
        result = []
        for pose in poses:
            seed = self.solve(pose, seed)
            result.append(seed.copy())
        return np.asarray(result, dtype=np.float32)
