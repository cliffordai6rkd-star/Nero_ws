from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from calibration.dynamics_common import DOF, DynamicsPlan, build_reduced_model
from calibration.identification import BaseParameterFit, PhysicalParameterFit


def write_identified_urdf(
    plan: DynamicsPlan,
    physical_fit: PhysicalParameterFit,
    base_fit: BaseParameterFit,
    *,
    output_path: str | Path | None = None,
    manifest_path: str | Path = "calibration/results/dynamics_manifest.yaml",
    training_data: list[str | Path] | None = None,
) -> tuple[Path, Path]:
    source = plan.model.urdf_path.resolve()
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source.with_name(f"{source.stem}_identified{source.suffix}")
    )
    if output == source:
        raise RuntimeError("refusing to overwrite the source URDF")
    if output.parent != source.parent:
        raise RuntimeError(
            "identified URDF must remain beside the source URDF so relative mesh paths stay valid"
        )
    if physical_fit.parameters.size != 10 * DOF + 3 * DOF:
        raise ValueError("physical fit has an unexpected parameter count")

    pin, reduced_model = build_reduced_model(plan.model)
    tree = ET.parse(source)
    root = tree.getroot()
    dynamic_parameters = physical_fit.parameters[: 10 * DOF].reshape(DOF, 10)
    child_links = [_joint_child_link(root, name) for name in plan.model.joint_names]

    # The reduced model collapses locked gripper bodies into joint7. Remove those
    # descendant inertias before writing the identified joint7 aggregate, otherwise
    # Pinocchio would count them twice when loading the generated URDF.
    terminal_link = child_links[-1]
    for link_name in _descendant_links(root, terminal_link).difference({terminal_link}):
        link = _find_link(root, link_name)
        inertial = link.find("inertial")
        if inertial is not None:
            link.remove(inertial)

    for link_name, parameters in zip(child_links, dynamic_parameters):
        inertia = _inertia_from_dynamic_parameters(pin, parameters)
        _set_link_inertial(
            _find_link(root, link_name),
            float(inertia.mass),
            np.asarray(inertia.lever, dtype=np.float64),
            np.asarray(inertia.inertia, dtype=np.float64),
        )
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)

    generated_settings = replace(plan.model, urdf_path=output)
    _, generated_model = build_reduced_model(generated_settings)
    generated_parameters = np.concatenate(
        [
            np.asarray(generated_model.inertias[joint_id].toDynamicParameters(), dtype=np.float64)
            for joint_id in range(1, DOF + 1)
        ]
    )
    if not np.allclose(
        generated_parameters,
        dynamic_parameters.reshape(-1),
        rtol=1e-6,
        atol=1e-9,
    ):
        raise RuntimeError(
            "generated URDF does not reproduce the recovered reduced-model inertias; "
            f"max_error={np.max(np.abs(generated_parameters - dynamic_parameters.reshape(-1))):.6g}"
        )

    manifest = Path(manifest_path).expanduser().resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_urdf": str(source),
        "identified_urdf": str(output),
        "config_path": str(plan.source_path),
        "training_data": [
            str(Path(path).expanduser().resolve()) for path in (training_data or [])
        ],
        "joint_names": list(plan.model.joint_names),
        "locked_joint_names": list(plan.model.locked_joint_names),
        "terminal_aggregate_link": terminal_link,
        "collapsed_descendant_inertias": sorted(
            _descendant_links(root, terminal_link).difference({terminal_link})
        ),
        "friction": {
            "coulomb_nm": physical_fit.coulomb_nm.tolist(),
            "viscous_nm_per_rad_s": physical_fit.viscous_nm_per_rad_s.tolist(),
            "coulomb_velocity_scale_rad_s": (
                plan.preprocess.coulomb_velocity_scale_rad_s
            ),
            "velocity_sign_model": (
                "tanh(dq / "
                f"{plan.preprocess.coulomb_velocity_scale_rad_s:.12g})"
            ),
        },
        "joint_torque_bias_nm": physical_fit.bias_nm.tolist(),
        "base_parameter_fit": {
            "rank": base_fit.rank,
            "condition_number": base_fit.condition_number,
            "singular_values": base_fit.singular_values.tolist(),
            "irls_iterations": base_fit.irls_iterations,
        },
        "physical_recovery": {
            "optimizer_success": physical_fit.optimizer_success,
            "optimizer_message": physical_fit.optimizer_message,
            "optimizer_cost": physical_fit.optimizer_cost,
            "optimizer_nfev": physical_fit.optimizer_nfev,
            "optimizer_optimality": physical_fit.optimizer_optimality,
            "optimizer_backend": physical_fit.optimizer_backend,
            "optimizer_device": physical_fit.optimizer_device,
            "inertia_eigenvalues_kg_m2": physical_fit.inertia_eigenvalues.tolist(),
        },
        "dynamic_parameters": dynamic_parameters.tolist(),
        "source_urdf_overwritten": False,
    }
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output, manifest


def load_identified_parameters(manifest_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(manifest_path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    friction = payload.get("friction", {})
    coulomb = np.asarray(friction.get("coulomb_nm"), dtype=np.float64).reshape(-1)
    viscous = np.asarray(friction.get("viscous_nm_per_rad_s"), dtype=np.float64).reshape(-1)
    bias = np.asarray(payload.get("joint_torque_bias_nm"), dtype=np.float64).reshape(-1)
    for name, value in (("coulomb", coulomb), ("viscous", viscous), ("bias", bias)):
        if value.size != DOF or not np.isfinite(value).all():
            raise ValueError(f"manifest contains invalid {name} parameters")
    return coulomb, viscous, bias


def _inertia_from_dynamic_parameters(pin, parameters):
    constructor = getattr(pin.Inertia, "FromDynamicParameters", None)
    if not callable(constructor):
        raise RuntimeError("Pinocchio 3.x Inertia.FromDynamicParameters is required")
    inertia = constructor(np.asarray(parameters, dtype=np.float64))
    eigenvalues = np.linalg.eigvalsh(np.asarray(inertia.inertia, dtype=np.float64))
    if inertia.mass <= 0 or np.any(eigenvalues <= 0):
        raise RuntimeError("refusing to write a non-physical link inertia")
    principal = np.sort(eigenvalues)
    if principal[2] > principal[0] + principal[1] + 1e-10:
        raise RuntimeError("refusing to write inertia violating the triangle inequality")
    return inertia


def _set_link_inertial(link, mass, com, inertia):
    old = link.find("inertial")
    if old is not None:
        link.remove(old)
    inertial = ET.Element("inertial")
    ET.SubElement(
        inertial,
        "origin",
        {"rpy": "0 0 0", "xyz": " ".join(_format(value) for value in com)},
    )
    ET.SubElement(inertial, "mass", {"value": _format(mass)})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": _format(inertia[0, 0]),
            "ixy": _format(inertia[0, 1]),
            "ixz": _format(inertia[0, 2]),
            "iyy": _format(inertia[1, 1]),
            "iyz": _format(inertia[1, 2]),
            "izz": _format(inertia[2, 2]),
        },
    )
    link.insert(0, inertial)


def _joint_child_link(root, joint_name):
    for joint in root.findall("joint"):
        if joint.get("name") == joint_name:
            child = joint.find("child")
            if child is not None and child.get("link"):
                return str(child.get("link"))
    raise ValueError(f"joint {joint_name!r} or its child link was not found in URDF")


def _descendant_links(root, start_link):
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None and parent.get("link") and child.get("link"):
            children.setdefault(str(parent.get("link")), []).append(str(child.get("link")))
    result = {start_link}
    pending = [start_link]
    while pending:
        parent = pending.pop()
        for child in children.get(parent, []):
            if child not in result:
                result.add(child)
                pending.append(child)
    return result


def _find_link(root, name):
    for link in root.findall("link"):
        if link.get("name") == name:
            return link
    raise ValueError(f"link {name!r} not found in URDF")


def _format(value):
    return f"{float(value):.12g}"
