from typing import Union
import functools

import numpy as np
import torch
from scipy.spatial.transform import Rotation

class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self, 
            from_rep='axis_angle', 
            to_rep='rotation_6d', 
            from_convention=None,
            to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != 'matrix':
            funcs = [
                getattr(pt, f'{from_rep}_to_matrix'),
                getattr(pt, f'matrix_to_{from_rep}')
            ]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != 'matrix':
            funcs = [
                getattr(pt, f'matrix_to_{to_rep}'),
                getattr(pt, f'{to_rep}_to_matrix')
            ]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])
        
        inverse_funcs = inverse_funcs[::-1]
        
        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _matrix_to_rotation_6d_numpy(matrix: np.ndarray) -> np.ndarray:
        return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)

    @staticmethod
    def _rotation_6d_to_matrix_numpy(rotation_6d: np.ndarray) -> np.ndarray:
        a1 = rotation_6d[..., 0:3]
        a2 = rotation_6d[..., 3:6]

        b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(min=1e-12)
        proj = np.sum(b1 * a2, axis=-1, keepdims=True)
        b2 = a2 - proj * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True).clip(min=1e-12)
        b3 = np.cross(b1, b2, axis=-1)

        return np.stack((b1, b2, b3), axis=-2)

    @staticmethod
    def _matrix_to_euler_numpy(matrix: np.ndarray, convention: str) -> np.ndarray:
        return Rotation.from_matrix(matrix).as_euler(convention.lower(), degrees=False)

    @staticmethod
    def _euler_to_matrix_numpy(euler: np.ndarray, convention: str) -> np.ndarray:
        return Rotation.from_euler(convention.lower(), euler, degrees=False).as_matrix()

    @staticmethod
    def _matrix_to_axis_angle_numpy(matrix: np.ndarray) -> np.ndarray:
        return Rotation.from_matrix(matrix).as_rotvec()

    @staticmethod
    def _axis_angle_to_matrix_numpy(axis_angle: np.ndarray) -> np.ndarray:
        return Rotation.from_rotvec(axis_angle).as_matrix()

    @staticmethod
    def _matrix_to_quaternion_numpy(matrix: np.ndarray) -> np.ndarray:
        quat_xyzw = Rotation.from_matrix(matrix).as_quat()
        return quat_xyzw[..., [3, 0, 1, 2]]

    @staticmethod
    def _quaternion_to_matrix_numpy(quaternion: np.ndarray) -> np.ndarray:
        quat_xyzw = quaternion[..., [1, 2, 3, 0]]
        return Rotation.from_quat(quat_xyzw).as_matrix()

    @classmethod
    def _get_numpy_funcs(cls, rep: str, convention=None):
        if rep == 'axis_angle':
            funcs = [cls._axis_angle_to_matrix_numpy, cls._matrix_to_axis_angle_numpy]
        elif rep == 'euler_angles':
            funcs = [
                functools.partial(cls._euler_to_matrix_numpy, convention=convention),
                functools.partial(cls._matrix_to_euler_numpy, convention=convention)
            ]
        elif rep == 'quaternion':
            funcs = [cls._quaternion_to_matrix_numpy, cls._matrix_to_quaternion_numpy]
        elif rep == 'rotation_6d':
            funcs = [cls._rotation_6d_to_matrix_numpy, cls._matrix_to_rotation_6d_numpy]
        elif rep == 'matrix':
            funcs = [lambda x: x, lambda x: x]
        else:
            raise ValueError(f'Unsupported rotation representation: {rep}')
        return funcs

    @classmethod
    def _build_funcs(cls, from_rep: str, to_rep: str, from_convention=None, to_convention=None):
        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != 'matrix':
            funcs = cls._get_numpy_funcs(from_rep, from_convention)
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != 'matrix':
            funcs = cls._get_numpy_funcs(to_rep, to_convention)
            forward_funcs.append(funcs[1])
            inverse_funcs.append(funcs[0])

        return forward_funcs, inverse_funcs[::-1]

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        tensor_meta = None
        if isinstance(x, torch.Tensor):
            tensor_meta = {
                'device': x.device,
                'dtype': x.dtype
            }
            x_ = x.detach().cpu().numpy()

        for func in funcs:
            x_ = func(x_)

        if tensor_meta is not None:
            return torch.as_tensor(x_, device=tensor_meta['device'], dtype=tensor_meta['dtype'])
        return x_
        
    def forward(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)
    
    def inverse(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


RotationTransformer._build_funcs  # silence lint-style unused warning


def _rotation_transformer_init(self, from_rep='axis_angle', to_rep='rotation_6d',
        from_convention=None, to_convention=None):
    assert from_rep != to_rep
    assert from_rep in self.valid_reps
    assert to_rep in self.valid_reps
    if from_rep == 'euler_angles':
        assert from_convention is not None
    if to_rep == 'euler_angles':
        assert to_convention is not None

    forward_funcs, inverse_funcs = self._build_funcs(
        from_rep=from_rep,
        to_rep=to_rep,
        from_convention=from_convention,
        to_convention=to_convention
    )
    self.forward_funcs = forward_funcs
    self.inverse_funcs = inverse_funcs


RotationTransformer.__init__ = _rotation_transformer_init


def test():
    tf = RotationTransformer()

    rotvec = np.random.uniform(-2*np.pi,2*np.pi,size=(1000,3))
    rot6d = tf.forward(rotvec)
    new_rotvec = tf.inverse(rot6d)

    from scipy.spatial.transform import Rotation
    diff = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(new_rotvec).inv()
    dist = diff.magnitude()
    assert dist.max() < 1e-7

    tf = RotationTransformer('rotation_6d', 'matrix')
    rot6d_wrong = rot6d + np.random.normal(scale=0.1, size=rot6d.shape)
    mat = tf.forward(rot6d_wrong)
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1)
    # rotaiton_6d will be normalized to rotation matrix
