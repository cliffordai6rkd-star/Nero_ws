from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from calibration.dynamics_common import DOF, IdentificationConfig
from calibration.regressor import PinocchioDynamicsRegressor, RegressorData


@dataclass(frozen=True)
class BaseParameterFit:
    parameters: np.ndarray
    rank: int
    singular_values: np.ndarray
    condition_number: float
    column_scale: np.ndarray
    identifiable_basis: np.ndarray
    irls_iterations: int
    robust_weights: np.ndarray


@dataclass(frozen=True)
class PhysicalParameterFit:
    parameters: np.ndarray
    coulomb_nm: np.ndarray
    viscous_nm_per_rad_s: np.ndarray
    bias_nm: np.ndarray
    optimizer_success: bool
    optimizer_message: str
    optimizer_cost: float
    optimizer_nfev: int
    optimizer_optimality: float
    optimizer_backend: str
    optimizer_device: str
    inertia_eigenvalues: np.ndarray


def fit_identifiable_base_parameters(
    regressor: RegressorData,
    config: IdentificationConfig,
) -> BaseParameterFit:
    matrix = np.asarray(regressor.matrix, dtype=np.float64)
    observation = np.asarray(regressor.observation, dtype=np.float64)
    prior = np.asarray(regressor.prior_parameters, dtype=np.float64)
    joint_index = np.tile(np.arange(DOF), matrix.shape[0] // DOF)
    nominal_residual = observation - matrix @ prior
    joint_scale = np.asarray(
        [_robust_scale(nominal_residual[joint_index == joint]) for joint in range(DOF)]
    )
    sample_weight = 1.0 / joint_scale[joint_index]
    weighted_matrix = matrix * sample_weight[:, None]
    weighted_target = nominal_residual * sample_weight

    column_scale = np.linalg.norm(weighted_matrix, axis=0)
    column_scale[column_scale <= np.finfo(np.float64).eps] = 1.0
    normalized = weighted_matrix / column_scale[None, :]
    _, singular_values, vt = np.linalg.svd(normalized, full_matrices=False)
    if not singular_values.size:
        raise RuntimeError("empty SVD while extracting identifiable parameters")
    threshold = singular_values[0] * config.svd_relative_tolerance
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank < 1:
        raise RuntimeError("dynamics regressor has no identifiable singular directions")
    basis = vt[:rank]
    design = normalized @ basis.T
    beta = np.zeros(rank, dtype=np.float64)
    robust_weights = np.ones(matrix.shape[0], dtype=np.float64)
    iterations = 0
    for iterations in range(1, config.max_irls_iterations + 1):
        sqrt_robust = np.sqrt(robust_weights)
        lhs = design * sqrt_robust[:, None]
        rhs = weighted_target * sqrt_robust
        normal = lhs.T @ lhs + config.ridge * np.eye(rank)
        candidate = np.linalg.solve(normal, lhs.T @ rhs)
        normalized_residual = weighted_target - design @ candidate
        updated_weights = _huber_weights(normalized_residual, config.huber_delta)
        if np.linalg.norm(candidate - beta) <= 1e-9 * (1.0 + np.linalg.norm(beta)):
            beta = candidate
            robust_weights = updated_weights
            break
        beta = candidate
        robust_weights = updated_weights

    prior_scaled = prior * column_scale
    fitted_scaled = prior_scaled + basis.T @ beta
    parameters = fitted_scaled / column_scale
    identifiable = singular_values[:rank]
    condition = float(identifiable[0] / identifiable[-1])
    return BaseParameterFit(
        parameters=parameters,
        rank=rank,
        singular_values=singular_values,
        condition_number=condition,
        column_scale=column_scale,
        identifiable_basis=basis,
        irls_iterations=iterations,
        robust_weights=robust_weights,
    )


def recover_physical_parameters(
    dynamics: PinocchioDynamicsRegressor,
    base_fit: BaseParameterFit,
    config: IdentificationConfig,
) -> PhysicalParameterFit:
    pin = dynamics.pin
    initial, prior_scale = _encode_prior(dynamics, config)
    lower, upper = _physical_bounds(initial, config)
    target_scaled = base_fit.parameters * base_fit.column_scale
    singular_weight = np.sqrt(
        base_fit.singular_values[: base_fit.rank] / base_fit.singular_values[0]
    )

    def residual(variables: np.ndarray) -> np.ndarray:
        parameters, _, _, _, _ = _decode_physical(pin, variables)
        identifiable_error = base_fit.identifiable_basis @ (
            parameters * base_fit.column_scale - target_scaled
        )
        prior_error = (variables - initial) / prior_scale
        return np.concatenate(
            (
                singular_weight * identifiable_error,
                np.sqrt(config.physical_prior_weight) * prior_error,
            )
        )

    residual_function = residual
    jacobian = "2-point"
    optimizer_backend = "scipy-numerical-jacobian"
    optimizer_device = "cpu"
    if config.physical_optimizer_backend == "jax":
        residual_function, jacobian, optimizer_device = _build_jax_physical_functions(
            base_fit,
            initial,
            prior_scale,
            config,
        )
        optimizer_backend = "jax-autodiff-jacobian"
        cpu_initial_residual = residual(initial)
        gpu_initial_residual = residual_function(initial)
        if not np.allclose(
            cpu_initial_residual,
            gpu_initial_residual,
            rtol=1e-9,
            atol=1e-10,
        ):
            raise RuntimeError(
                "JAX physical residual does not match the Pinocchio/NumPy reference "
                f"at initialization; max_error="
                f"{np.max(np.abs(cpu_initial_residual - gpu_initial_residual)):.6g}"
            )

    result = least_squares(
        residual_function,
        np.clip(initial, lower + 1e-12, upper - 1e-12),
        jac=jacobian,
        bounds=(lower, upper),
        loss="huber",
        f_scale=config.huber_delta,
        max_nfev=config.max_physical_evaluations,
        ftol=config.physical_tolerance,
        xtol=config.physical_tolerance,
        gtol=config.physical_tolerance,
        x_scale=prior_scale,
    )
    parameters, coulomb, viscous, bias, eigenvalues = _decode_physical(pin, result.x)
    _validate_physical(parameters, eigenvalues, config)
    return PhysicalParameterFit(
        parameters=parameters,
        coulomb_nm=coulomb,
        viscous_nm_per_rad_s=viscous,
        bias_nm=bias,
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        optimizer_cost=float(result.cost),
        optimizer_nfev=int(result.nfev),
        optimizer_optimality=float(result.optimality),
        optimizer_backend=optimizer_backend,
        optimizer_device=optimizer_device,
        inertia_eigenvalues=eigenvalues,
    )


def _build_jax_physical_functions(base_fit, initial, prior_scale, config):
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError(
            "physical_optimizer_backend=jax requires the GPU extra; install with "
            "python -m pip install -e '.[gpu-identification]'"
        ) from exc

    jax.config.update("jax_enable_x64", True)
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not gpu_devices:
        raise RuntimeError(
            "physical_optimizer_backend=jax requires a JAX CUDA device, but JAX did not "
            "detect one; check jax.devices() and the NVIDIA driver"
        )
    device = gpu_devices[0]
    basis = jax.device_put(jnp.asarray(base_fit.identifiable_basis), device)
    column_scale = jax.device_put(jnp.asarray(base_fit.column_scale), device)
    target_scaled = jax.device_put(
        jnp.asarray(base_fit.parameters * base_fit.column_scale), device
    )
    singular_weight = jax.device_put(
        jnp.asarray(
            np.sqrt(
                base_fit.singular_values[: base_fit.rank]
                / base_fit.singular_values[0]
            )
        ),
        device,
    )
    initial_device = jax.device_put(jnp.asarray(initial), device)
    prior_scale_device = jax.device_put(jnp.asarray(prior_scale), device)
    prior_weight = float(np.sqrt(config.physical_prior_weight))

    def decode_link(block):
        mass = jnp.exp(block[0])
        com = block[1:4]
        zero = jnp.asarray(0.0, dtype=block.dtype)
        chol = jnp.stack(
            (
                jnp.stack((jnp.exp(block[4]), zero, zero)),
                jnp.stack((block[5], jnp.exp(block[6]), zero)),
                jnp.stack((block[7], block[8], jnp.exp(block[9]))),
            )
        )
        second_moment = chol @ chol.T
        inertia_com = jnp.trace(second_moment) * jnp.eye(3) - second_moment
        inertia_origin = inertia_com + mass * (
            jnp.dot(com, com) * jnp.eye(3) - jnp.outer(com, com)
        )
        return jnp.stack(
            (
                mass,
                mass * com[0],
                mass * com[1],
                mass * com[2],
                inertia_origin[0, 0],
                inertia_origin[0, 1],
                inertia_origin[1, 1],
                inertia_origin[0, 2],
                inertia_origin[1, 2],
                inertia_origin[2, 2],
            )
        )

    def residual_device(variables):
        inertial = jax.vmap(decode_link)(variables[: 10 * DOF].reshape(DOF, 10))
        parameters = jnp.concatenate((inertial.reshape(-1), variables[10 * DOF :]))
        identifiable_error = basis @ (parameters * column_scale - target_scaled)
        prior_error = (variables - initial_device) / prior_scale_device
        return jnp.concatenate(
            (singular_weight * identifiable_error, prior_weight * prior_error)
        )

    residual_jit = jax.jit(residual_device, device=device)
    jacobian_jit = jax.jit(jax.jacfwd(residual_device), device=device)
    residual_jit(initial_device).block_until_ready()
    jacobian_jit(initial_device).block_until_ready()

    def residual_numpy(variables):
        values = jax.device_put(jnp.asarray(variables), device)
        return np.array(
            jax.device_get(residual_jit(values)), dtype=np.float64, copy=True
        )

    def jacobian_numpy(variables):
        values = jax.device_put(jnp.asarray(variables), device)
        return np.array(
            jax.device_get(jacobian_jit(values)), dtype=np.float64, copy=True
        )

    return residual_numpy, jacobian_numpy, str(device)


def _encode_prior(dynamics, config):
    values: list[float] = []
    scales: list[float] = []
    for joint_id in range(1, DOF + 1):
        inertia = dynamics.model.inertias[joint_id]
        mass = max(float(inertia.mass), config.mass_bounds_kg[0])
        com = np.asarray(inertia.lever, dtype=np.float64)
        inertia_com = np.asarray(inertia.inertia, dtype=np.float64)
        second_moment = 0.5 * np.trace(inertia_com) * np.eye(3) - inertia_com
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (second_moment + second_moment.T))
        second_moment = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-12)) @ eigenvectors.T
        chol = np.linalg.cholesky(second_moment + 1e-12 * np.eye(3))
        values.extend(
            [
                np.log(mass),
                *com,
                np.log(chol[0, 0]),
                chol[1, 0],
                np.log(chol[1, 1]),
                chol[2, 0],
                chol[2, 1],
                np.log(chol[2, 2]),
            ]
        )
        scales.extend([1.0, 0.1, 0.1, 0.1, 1.0, 0.1, 1.0, 0.1, 0.1, 1.0])
    values.extend([0.0] * (3 * DOF))
    scales.extend(config.max_coulomb_nm.tolist())
    scales.extend(config.max_viscous_nm_per_rad_s.tolist())
    scales.extend(config.max_abs_bias_nm.tolist())
    return np.asarray(values), np.maximum(np.asarray(scales), 1e-9)


def _physical_bounds(initial, config):
    lower: list[float] = []
    upper: list[float] = []
    log_mass_lower = np.log(config.mass_bounds_kg[0])
    log_mass_upper = np.log(config.mass_bounds_kg[1])
    for _ in range(DOF):
        lower.extend(
            [log_mass_lower, *([-config.max_abs_com_m] * 3), -16.0, -2.0, -16.0, -2.0, -2.0, -16.0]
        )
        upper.extend(
            [log_mass_upper, *([config.max_abs_com_m] * 3), 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        )
    lower.extend([0.0] * DOF)
    upper.extend(config.max_coulomb_nm.tolist())
    lower.extend([0.0] * DOF)
    upper.extend(config.max_viscous_nm_per_rad_s.tolist())
    lower.extend((-config.max_abs_bias_nm).tolist())
    upper.extend(config.max_abs_bias_nm.tolist())
    result_lower = np.asarray(lower, dtype=np.float64)
    result_upper = np.asarray(upper, dtype=np.float64)
    if result_lower.shape != initial.shape:
        raise RuntimeError("internal physical-parameter bound size mismatch")
    return result_lower, result_upper


def _decode_physical(pin, variables):
    variables = np.asarray(variables, dtype=np.float64)
    inertial_parameters: list[np.ndarray] = []
    eigenvalues: list[np.ndarray] = []
    for joint in range(DOF):
        offset = 10 * joint
        block = variables[offset : offset + 10]
        mass = float(np.exp(block[0]))
        com = block[1:4]
        chol = np.asarray(
            [
                [np.exp(block[4]), 0.0, 0.0],
                [block[5], np.exp(block[6]), 0.0],
                [block[7], block[8], np.exp(block[9])],
            ]
        )
        second_moment = chol @ chol.T
        inertia_com = np.trace(second_moment) * np.eye(3) - second_moment
        inertia = pin.Inertia(mass, com, inertia_com)
        inertial_parameters.append(
            np.asarray(inertia.toDynamicParameters(), dtype=np.float64)
        )
        eigenvalues.append(np.linalg.eigvalsh(inertia_com))
    friction_offset = 10 * DOF
    coulomb = variables[friction_offset : friction_offset + DOF].copy()
    viscous = variables[friction_offset + DOF : friction_offset + 2 * DOF].copy()
    bias = variables[friction_offset + 2 * DOF : friction_offset + 3 * DOF].copy()
    parameters = np.concatenate((*inertial_parameters, coulomb, viscous, bias))
    return parameters, coulomb, viscous, bias, np.stack(eigenvalues)


def _validate_physical(parameters, inertia_eigenvalues, config):
    if not np.isfinite(parameters).all() or not np.isfinite(inertia_eigenvalues).all():
        raise RuntimeError("physical recovery produced non-finite parameters")
    if np.any(inertia_eigenvalues <= 0):
        raise RuntimeError(f"physical recovery produced non-positive inertia: {inertia_eigenvalues}")
    sorted_values = np.sort(inertia_eigenvalues, axis=1)
    if np.any(sorted_values[:, 2] > sorted_values[:, 0] + sorted_values[:, 1] + 1e-10):
        raise RuntimeError("physical recovery produced inertia violating the triangle inequality")
    masses = parameters[: 10 * DOF].reshape(DOF, 10)[:, 0]
    if np.any(masses < config.mass_bounds_kg[0]) or np.any(masses > config.mass_bounds_kg[1]):
        raise RuntimeError(f"physical recovery produced masses outside bounds: {masses}")


def _robust_scale(values):
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))
    return max(float(scale), 1e-3)


def _huber_weights(residual, delta):
    absolute = np.abs(residual)
    result = np.ones_like(absolute)
    mask = absolute > delta
    result[mask] = delta / absolute[mask]
    return result
