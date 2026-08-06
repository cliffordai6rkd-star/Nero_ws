import torch


def normalize_quaternion_xyzw(quaternion: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have last dimension 4")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    identity = torch.zeros_like(quaternion)
    identity[..., 3] = 1.0
    return torch.where(norm > eps, quaternion / norm.clamp_min(eps), identity)


def quaternion_multiply_xyzw(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lx, ly, lz, lw = lhs.unbind(dim=-1)
    rx, ry, rz, rw = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def relative_pose_to_absolute_pose(
    relative_pose: torch.Tensor,
    reference_pose: torch.Tensor,
) -> torch.Tensor:
    if reference_pose.ndim == relative_pose.ndim - 1:
        reference_pose = reference_pose.unsqueeze(-2)
    position = reference_pose[..., :3] + relative_pose[..., :3]
    reference_quaternion = normalize_quaternion_xyzw(reference_pose[..., 3:7])
    relative_quaternion = normalize_quaternion_xyzw(relative_pose[..., 3:7])
    quaternion = normalize_quaternion_xyzw(
        quaternion_multiply_xyzw(reference_quaternion, relative_quaternion)
    )
    return torch.cat((position, quaternion), dim=-1)
