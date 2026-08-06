from __future__ import annotations

import copy
from contextlib import contextmanager
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from calibration.dynamics_common import DOF, DynamicsPlan
from calibration.excitation import FourierTrajectory, validate_trajectory


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContactEvent:
    kind: str
    first_sample: int
    first_time_s: float
    last_sample: int
    last_time_s: float
    hit_sample_count: int
    minimum_distance_m: float
    geometry_a: str
    geometry_b: str


@dataclass(frozen=True)
class MujocoPreview:
    model: object
    qpos: np.ndarray
    end_effector_path_m: np.ndarray
    workspace_violation_count: int
    contact_events: tuple[ContactEvent, ...]
    contact_samples_checked: int
    scene_path: Path


@dataclass(frozen=True)
class MujocoLivePreview:
    model: object
    initial_qpos: np.ndarray
    joint_qpos_addresses: np.ndarray
    scene_path: Path


class MujocoPoseSafetyChecker:
    """Evaluate joint poses and interpolated legs against the configured MuJoCo scene."""

    def __init__(self, plan) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("automatic trajectory safety filtering requires mujoco") from exc
        robot_root = _convert_urdf_to_mjcf(
            mujoco,
            plan.model.urdf_path,
            plan.model.locked_joint_names,
        )
        scene_root = _merge_robot_into_scene(plan, robot_root)
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(
            ET.tostring(scene_root, encoding="unicode")
        )
        self.data = mujoco.MjData(self.model)
        self.base_qpos = np.asarray(self.data.qpos, dtype=np.float64).copy()
        self.addresses = _joint_qpos_addresses(mujoco, self.model, plan)
        self.body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            plan.simulation.end_effector_body,
        )
        if self.body_id < 0:
            raise ValueError(
                "MuJoCo model is missing end-effector body: "
                f"{plan.simulation.end_effector_body}"
            )
        self.workspace_min = np.asarray(plan.simulation.workspace_min_m, dtype=np.float64)
        self.workspace_max = np.asarray(plan.simulation.workspace_max_m, dtype=np.float64)
        self.ignored_contacts = set(plan.simulation.ignored_contact_pairs)

    def safe_mask(self, q: np.ndarray) -> np.ndarray:
        poses = np.asarray(q, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != DOF or not np.isfinite(poses).all():
            raise ValueError("MuJoCo safety checker requires finite joint poses with shape (N, 7)")
        result = np.ones(poses.shape[0], dtype=bool)
        timestamp = np.zeros(poses.shape[0], dtype=np.float64)
        for index, pose in enumerate(poses):
            self.data.qpos[:] = self.base_qpos
            self.data.qpos[self.addresses] = pose
            self.data.qvel[:] = 0.0
            self.mujoco.mj_forward(self.model, self.data)
            position = np.asarray(self.data.xpos[self.body_id], dtype=np.float64)
            if np.any(position < self.workspace_min) or np.any(position > self.workspace_max):
                result[index] = False
                continue
            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]
                event = _classify_contact(
                    self.mujoco,
                    self.model,
                    int(contact.geom1),
                    int(contact.geom2),
                    float(contact.dist),
                    index,
                    timestamp,
                    self.ignored_contacts,
                )
                if event is not None:
                    result[index] = False
                    break
        return result

    def leg_is_safe(self, start: np.ndarray, target: np.ndarray, samples: int = 25) -> bool:
        phase = np.linspace(0.0, 1.0, max(2, int(samples)), dtype=np.float64)
        first = np.asarray(start, dtype=np.float64)
        last = np.asarray(target, dtype=np.float64)
        poses = first[None, :] + phase[:, None] * (last - first)[None, :]
        return bool(np.all(self.safe_mask(poses)))


def prepare_mujoco_live_preview(
    plan: DynamicsPlan,
    initial_q: np.ndarray,
    scene_path: str | Path,
) -> MujocoLivePreview:
    log.info("building MuJoCo live preview scene initial_q=%s", np.array2string(np.asarray(initial_q), precision=4))
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo live preview requires mujoco>=3.3,<4") from exc
    q = np.asarray(initial_q, dtype=np.float64).reshape(-1)
    if q.shape != (DOF,) or not np.isfinite(q).all():
        raise ValueError("MuJoCo live preview requires a finite 7D initial pose")
    robot_root = _convert_urdf_to_mjcf(
        mujoco,
        plan.model.urdf_path,
        plan.model.locked_joint_names,
    )
    scene_root = _merge_robot_into_scene(plan, robot_root)
    output = Path(scene_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(scene_root, space="  ")
    ET.ElementTree(scene_root).write(output, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    addresses = _joint_qpos_addresses(mujoco, model, plan)
    data = mujoco.MjData(model)
    initial_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    initial_qpos[addresses] = q
    preview = MujocoLivePreview(
        model=model,
        initial_qpos=initial_qpos,
        joint_qpos_addresses=addresses,
        scene_path=output,
    )
    log.info(
        "MuJoCo live preview ready scene=%s qpos_addresses=%s",
        output,
        addresses.tolist(),
    )
    return preview


@contextmanager
def launch_mujoco_live_preview(preview: MujocoLivePreview):
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("MuJoCo live preview requires mujoco>=3.3,<4") from exc
    _prefer_x11_for_wayland_glfw()
    data = mujoco.MjData(preview.model)
    data.qpos[:] = preview.initial_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(preview.model, data)
    log.info("opening MuJoCo live viewer scene=%s", preview.scene_path)
    with mujoco.viewer.launch_passive(
        preview.model,
        data,
        show_left_ui=True,
        show_right_ui=False,
    ) as viewer:
        viewer.sync()
        log.info("MuJoCo live viewer opened")
        try:
            yield data, viewer
        finally:
            log.info("MuJoCo live viewer closing")


def sync_mujoco_live_preview(preview, data, viewer, q: np.ndarray) -> np.ndarray:
    import mujoco

    values = np.asarray(q, dtype=np.float64).reshape(-1)
    if values.shape != (DOF,) or not np.isfinite(values).all():
        raise ValueError("MuJoCo live preview requires a finite 7D pose")
    with viewer.lock():
        data.qpos[preview.joint_qpos_addresses] = values
        data.qvel[:] = 0.0
        mujoco.mj_forward(preview.model, data)
        mapped = np.asarray(
            data.qpos[preview.joint_qpos_addresses],
            dtype=np.float64,
        ).copy()
    viewer.sync()
    return mapped


def prepare_mujoco_preview(
    plan: DynamicsPlan,
    trajectory: FourierTrajectory,
    scene_path: str | Path,
) -> MujocoPreview:
    validate_trajectory(trajectory, plan)
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo trajectory playback requires mujoco>=3.3,<4"
        ) from exc

    robot_root = _convert_urdf_to_mjcf(
        mujoco,
        plan.model.urdf_path,
        plan.model.locked_joint_names,
    )
    scene_root = _merge_robot_into_scene(plan, robot_root)
    preliminary_xml = ET.tostring(scene_root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(preliminary_xml)
    qpos = _build_qpos_trajectory(mujoco, model, trajectory, plan)
    end_effector_path, contact_events, contact_samples_checked = _scan_trajectory(
        mujoco, model, qpos, trajectory.time_s, plan
    )
    workspace = plan.simulation
    outside = np.any(
        (end_effector_path < workspace.workspace_min_m[None, :])
        | (end_effector_path > workspace.workspace_max_m[None, :]),
        axis=1,
    )
    violation_count = int(np.count_nonzero(outside))
    _add_end_effector_path(scene_root, end_effector_path, violation_count > 0)

    output = Path(scene_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(scene_root, space="  ")
    ET.ElementTree(scene_root).write(output, encoding="utf-8", xml_declaration=True)

    final_model = mujoco.MjModel.from_xml_path(str(output))
    final_qpos = _build_qpos_trajectory(mujoco, final_model, trajectory, plan)
    return MujocoPreview(
        model=final_model,
        qpos=final_qpos,
        end_effector_path_m=end_effector_path,
        workspace_violation_count=violation_count,
        contact_events=contact_events,
        contact_samples_checked=contact_samples_checked,
        scene_path=output,
    )


def play_mujoco_preview(
    plan: DynamicsPlan,
    trajectory: FourierTrajectory,
    preview: MujocoPreview,
    *,
    playback_speed: float | None = None,
    loops: int = 1,
    hold_seconds: float = 3.0,
) -> None:
    if loops < 1:
        raise ValueError("simulation loops must be at least one")
    speed = plan.simulation.playback_speed if playback_speed is None else float(playback_speed)
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("playback speed must be positive and finite")
    if hold_seconds < 0:
        raise ValueError("hold seconds must be non-negative")
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo native viewer requires mujoco>=3.3,<4") from exc
    _prefer_x11_for_wayland_glfw()
    try:
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("MuJoCo native viewer requires mujoco>=3.3,<4") from exc

    data = mujoco.MjData(preview.model)
    source_rate = plan.excitation.sample_rate_hz
    display_rate = min(plan.simulation.display_rate_hz, source_rate)
    # Keep viewer work bounded at high playback speeds. Collision/workspace checks
    # already ran over the configured trajectory before visualization.
    stride = max(1, int(round(source_rate * speed / display_rate)))
    indices = np.arange(0, trajectory.time_s.size, stride, dtype=int)
    if indices[-1] != trajectory.time_s.size - 1:
        indices = np.append(indices, trajectory.time_s.size - 1)

    with mujoco.viewer.launch_passive(
        preview.model,
        data,
        show_left_ui=True,
        show_right_ui=False,
    ) as viewer:
        for _ in range(loops):
            start = time.monotonic()
            first_time = float(trajectory.time_s[indices[0]])
            for index in indices:
                if not viewer.is_running():
                    return
                deadline = start + (float(trajectory.time_s[index]) - first_time) / speed
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                data.qpos[:] = preview.qpos[index]
                data.qvel[:] = 0.0
                mujoco.mj_forward(preview.model, data)
                viewer.sync()
        deadline = time.monotonic() + hold_seconds
        while viewer.is_running() and time.monotonic() < deadline:
            viewer.sync()
            time.sleep(0.02)


def print_preview_report(
    plan: DynamicsPlan,
    trajectory: FourierTrajectory,
    preview: MujocoPreview,
) -> None:
    path = preview.end_effector_path_m
    print(f"MuJoCo scene: {preview.scene_path}")
    print(
        "end-effector xyz min [m]: "
        + np.array2string(np.min(path, axis=0), precision=4)
    )
    print(
        "end-effector xyz max [m]: "
        + np.array2string(np.max(path, axis=0), precision=4)
    )
    print(
        "workspace bounds [m]: "
        f"min={plan.simulation.workspace_min_m.tolist()} "
        f"max={plan.simulation.workspace_max_m.tolist()}"
    )
    print(f"workspace violations: {preview.workspace_violation_count}/{trajectory.time_s.size}")
    self_contacts = [event for event in preview.contact_events if event.kind == "self"]
    world_contacts = [event for event in preview.contact_events if event.kind == "world"]
    print(f"non-neighbor self-contact pairs: {len(self_contacts)}")
    print(f"unexpected world-contact pairs: {len(world_contacts)}")
    print(f"configured ignored contact pairs: {len(plan.simulation.ignored_contact_pairs)}")
    for first, second in plan.simulation.ignored_contact_pairs:
        print(f"  IGNORED_CONTACT {first} <-> {second}")
    print(f"contact samples checked: {preview.contact_samples_checked}")
    for event in preview.contact_events:
        print(
            f"  {event.kind.upper()}_CONTACT "
            f"first={event.first_time_s:.3f}s/{event.first_sample} "
            f"last={event.last_time_s:.3f}s/{event.last_sample} "
            f"hits={event.hit_sample_count}/{preview.contact_samples_checked} "
            f"min_distance={event.minimum_distance_m:.6g}m "
            f"{event.geometry_a} <-> {event.geometry_b}"
        )


def _convert_urdf_to_mjcf(
    mujoco,
    urdf_path: Path,
    locked_joint_names: tuple[str, ...],
) -> ET.Element:
    source = Path(urdf_path).resolve()
    tree = ET.parse(source)
    robot = tree.getroot()
    _lock_urdf_joints_for_preview(robot, locked_joint_names)
    for mesh in robot.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = (source.parent / mesh_path).resolve()
        if not mesh_path.is_file():
            raise ValueError(f"URDF mesh does not exist: {mesh_path}")
        mesh.set("filename", str(mesh_path))
    extension = robot.find("mujoco")
    if extension is None:
        extension = ET.SubElement(robot, "mujoco")
    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    # Nero visual assets are Collada DAE, which MuJoCo does not import. Keep the
    # STL collision meshes; they are the geometry used for preflight contacts.
    compiler.set("discardvisual", "true")
    compiler.set("fusestatic", "false")
    compiler.set("strippath", "false")

    with tempfile.TemporaryDirectory(prefix="nero_mujoco_") as temporary:
        temporary_path = Path(temporary)
        urdf_copy = temporary_path / "nero_for_mujoco.urdf"
        mjcf_copy = temporary_path / "nero_robot.xml"
        tree.write(urdf_copy, encoding="utf-8", xml_declaration=True)
        robot_model = mujoco.MjModel.from_xml_path(str(urdf_copy))
        mujoco.mj_saveLastXML(str(mjcf_copy), robot_model)
        mjcf_root = ET.parse(mjcf_copy).getroot()
    _make_asset_paths_absolute(mjcf_root, source.parent)
    _name_mjcf_geometries(mjcf_root)
    return mjcf_root


def _lock_urdf_joints_for_preview(
    robot: ET.Element,
    locked_joint_names: tuple[str, ...],
) -> None:
    joints = {
        str(joint.get("name")): joint
        for joint in robot.findall("joint")
        if joint.get("name")
    }
    missing = sorted(set(locked_joint_names).difference(joints))
    if missing:
        raise ValueError(f"locked joints not found while building MuJoCo preview: {missing}")
    for name in locked_joint_names:
        joint = joints[name]
        joint.set("type", "fixed")
        for child_name in (
            "axis",
            "limit",
            "mimic",
            "dynamics",
            "safety_controller",
            "calibration",
        ):
            child = joint.find(child_name)
            if child is not None:
                joint.remove(child)


def _make_asset_paths_absolute(root: ET.Element, urdf_directory: Path) -> None:
    for asset in root.findall("./asset/*"):
        filename = asset.get("file")
        if not filename or Path(filename).is_absolute():
            continue
        candidates = (
            urdf_directory / filename,
            urdf_directory / "meshes" / filename,
            urdf_directory / "meshes" / "dae" / filename,
        )
        resolved = next((path.resolve() for path in candidates if path.is_file()), None)
        if resolved is None:
            raise ValueError(f"could not resolve MuJoCo asset {filename!r} from {urdf_directory}")
        asset.set("file", str(resolved))


def _name_mjcf_geometries(root: ET.Element) -> None:
    world = root.find("worldbody")
    if world is None:
        raise ValueError("converted MuJoCo robot is missing worldbody")
    counter = 0

    def visit(parent: ET.Element, body_name: str) -> None:
        nonlocal counter
        local_index = 0
        for geom in parent.findall("geom"):
            if not geom.get("name"):
                geom.set("name", f"{body_name}_collision_{local_index}")
            local_index += 1
            counter += 1
        for body in parent.findall("body"):
            visit(body, str(body.get("name") or f"body_{counter}"))

    visit(world, "world")


def _prefer_x11_for_wayland_glfw() -> None:
    if os.environ.get("XDG_SESSION_TYPE", "").lower() != "wayland":
        return
    if not os.environ.get("DISPLAY"):
        return
    os.environ.setdefault("GLFW_PLATFORM", "x11")
    try:
        import glfw

        platform_hint = getattr(glfw, "PLATFORM", None)
        x11_platform = getattr(glfw, "PLATFORM_X11", None)
        if platform_hint is not None and x11_platform is not None:
            glfw.init_hint(platform_hint, x11_platform)
    except Exception:
        # The environment variable still gives GLFW 3.4+ a backend preference.
        pass


def _merge_robot_into_scene(plan: DynamicsPlan, robot_root: ET.Element) -> ET.Element:
    scene_root = copy.deepcopy(ET.parse(plan.simulation.scene_template_path).getroot())
    option = scene_root.find("option")
    if option is None:
        option = ET.SubElement(scene_root, "option")
    option.set("gravity", " ".join(f"{value:.12g}" for value in plan.model.gravity_m_s2))
    scene_asset = scene_root.find("asset")
    if scene_asset is None:
        scene_asset = ET.SubElement(scene_root, "asset")
    robot_asset = robot_root.find("asset")
    if robot_asset is not None:
        for item in robot_asset:
            scene_asset.append(copy.deepcopy(item))

    scene_world = scene_root.find("worldbody")
    robot_world = robot_root.find("worldbody")
    if scene_world is None or robot_world is None:
        raise ValueError("MuJoCo scene and converted robot must both contain worldbody")
    for item in robot_world:
        scene_world.append(copy.deepcopy(item))

    ignored = {"compiler", "option", "statistic", "visual", "asset", "worldbody"}
    for section in robot_root:
        if section.tag not in ignored:
            scene_root.append(copy.deepcopy(section))
    _configure_world_scene(scene_root, plan)
    return scene_root


def _configure_world_scene(root: ET.Element, plan: DynamicsPlan) -> None:
    world = root.find("worldbody")
    if world is None:
        raise ValueError("MuJoCo scene template is missing worldbody")
    floor = _find_named(world, "geom", "world_floor")
    floor.set("pos", f"0 0 {plan.simulation.floor_z_m:.12g}")


def _add_end_effector_path(root: ET.Element, path: np.ndarray, violation: bool) -> None:
    world = root.find("worldbody")
    if world is None:
        raise ValueError("MuJoCo scene is missing worldbody")
    stride = max(1, path.shape[0] // 80)
    color = "0.9 0.15 0.12 0.9" if violation else "1 0.58 0.05 0.9"
    for marker_index, point in enumerate(path[::stride]):
        ET.SubElement(
            world,
            "geom",
            {
                "name": f"world_ee_path_{marker_index}",
                "type": "sphere",
                "pos": " ".join(f"{value:.12g}" for value in point),
                "size": "0.006",
                "rgba": color,
                "contype": "0",
                "conaffinity": "0",
                "group": "4",
            },
        )


def _build_qpos_trajectory(mujoco, model, trajectory, plan):
    data = mujoco.MjData(model)
    qpos = np.tile(np.asarray(data.qpos, dtype=np.float64), (trajectory.q.shape[0], 1))
    addresses = _joint_qpos_addresses(mujoco, model, plan)
    qpos[:, addresses] = trajectory.q
    return qpos


def _joint_qpos_addresses(mujoco, model, plan):
    addresses: list[int] = []
    for joint_name in plan.model.joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing arm joint: {joint_name}")
        if int(model.jnt_type[joint_id]) not in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }:
            raise ValueError(f"MuJoCo arm joint must be scalar: {joint_name}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(addresses, dtype=np.int32)


def _scan_trajectory(mujoco, model, qpos, time_s, plan):
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, plan.simulation.end_effector_body
    )
    if body_id < 0:
        raise ValueError(
            f"MuJoCo model is missing end-effector body: {plan.simulation.end_effector_body}"
        )
    data = mujoco.MjData(model)
    path = np.empty((qpos.shape[0], 3), dtype=np.float64)
    aggregates: dict[tuple[str, str, str], dict[str, object]] = {}
    stride = plan.simulation.collision_sample_stride
    checked_samples = 0
    for sample_index, values in enumerate(qpos):
        data.qpos[:] = values
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        path[sample_index] = data.xpos[body_id]
        if sample_index % stride != 0 and sample_index != qpos.shape[0] - 1:
            continue
        checked_samples += 1
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            event = _classify_contact(
                mujoco,
                model,
                int(contact.geom1),
                int(contact.geom2),
                float(contact.dist),
                sample_index,
                time_s,
                set(plan.simulation.ignored_contact_pairs),
            )
            if event is None:
                continue
            key = (event.kind, event.geometry_a, event.geometry_b)
            aggregate = aggregates.get(key)
            if aggregate is None:
                aggregates[key] = {
                    "event": event,
                    "last_sample": sample_index,
                    "last_time_s": float(time_s[sample_index]),
                    "samples": {sample_index},
                    "minimum_distance_m": event.minimum_distance_m,
                }
            else:
                aggregate["last_sample"] = sample_index
                aggregate["last_time_s"] = float(time_s[sample_index])
                samples = aggregate["samples"]
                assert isinstance(samples, set)
                samples.add(sample_index)
                aggregate["minimum_distance_m"] = min(
                    float(aggregate["minimum_distance_m"]),
                    event.minimum_distance_m,
                )
    events: list[ContactEvent] = []
    for aggregate in aggregates.values():
        first = aggregate["event"]
        assert isinstance(first, ContactEvent)
        samples = aggregate["samples"]
        assert isinstance(samples, set)
        events.append(
            ContactEvent(
                kind=first.kind,
                first_sample=first.first_sample,
                first_time_s=first.first_time_s,
                last_sample=int(aggregate["last_sample"]),
                last_time_s=float(aggregate["last_time_s"]),
                hit_sample_count=len(samples),
                minimum_distance_m=float(aggregate["minimum_distance_m"]),
                geometry_a=first.geometry_a,
                geometry_b=first.geometry_b,
            )
        )
    return path, tuple(events), checked_samples


def _classify_contact(
    mujoco,
    model,
    geom_a,
    geom_b,
    distance_m,
    sample_index,
    time_s,
    ignored_contact_pairs,
):
    name_a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a) or f"geom_{geom_a}"
    name_b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b) or f"geom_{geom_b}"
    if tuple(sorted((str(name_a), str(name_b)))) in ignored_contact_pairs:
        return None
    body_a = int(model.geom_bodyid[geom_a])
    body_b = int(model.geom_bodyid[geom_b])
    floor_a = name_a == "world_floor"
    floor_b = name_b == "world_floor"
    if floor_a or floor_b:
        robot_body = body_b if floor_a else body_a
        robot_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, robot_body) or "world"
        )
        if robot_name in {"base_link", "world", "world_scene"}:
            return None
        kind = "world"
    elif body_a == body_b or _body_graph_distance(model, body_a, body_b) <= 2:
        return None
    else:
        kind = "self"
    return ContactEvent(
        kind=kind,
        first_sample=int(sample_index),
        first_time_s=float(time_s[sample_index]),
        last_sample=int(sample_index),
        last_time_s=float(time_s[sample_index]),
        hit_sample_count=1,
        minimum_distance_m=float(distance_m),
        geometry_a=str(name_a),
        geometry_b=str(name_b),
    )


def _body_graph_distance(model, first, second):
    ancestors: dict[int, int] = {}
    current = int(first)
    distance = 0
    while True:
        ancestors[current] = distance
        if current == 0:
            break
        current = int(model.body_parentid[current])
        distance += 1
    current = int(second)
    distance = 0
    while current not in ancestors:
        current = int(model.body_parentid[current])
        distance += 1
    return distance + ancestors[current]


def _find_named(parent: ET.Element, tag: str, name: str) -> ET.Element:
    for element in parent.findall(tag):
        if element.get("name") == name:
            return element
    raise ValueError(f"MuJoCo scene is missing {tag} named {name!r}")
