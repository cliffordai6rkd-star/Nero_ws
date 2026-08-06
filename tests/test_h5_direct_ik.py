from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.infer_h5_direct_ik import (
    _frame_indices,
    _observation_image_indices,
    _select_arm_matrix,
    _sync_mujoco_action_visualization,
    _validate_visualization_inputs,
    nearest_indices,
    resolve_episode,
)


def test_nearest_indices_uses_earlier_sample_on_equal_distance() -> None:
    source = np.array([100, 200, 300], dtype=np.int64)
    targets = np.array([50, 150, 260, 400], dtype=np.int64)

    np.testing.assert_array_equal(
        nearest_indices(source, targets),
        [0, 0, 2, 2],
    )


def test_select_arm_matrix_supports_concatenated_joint_layout() -> None:
    values = np.arange(3 * 14, dtype=np.float64).reshape(3, 14)

    selected = _select_arm_matrix(
        values,
        count=3,
        width=7,
        arm_names=("left", "right"),
        arm_index=1,
        name="teleop/q_follower",
    )

    np.testing.assert_array_equal(selected, values[:, 7:14])


def test_frame_indices_applies_range_stride_and_maximum() -> None:
    np.testing.assert_array_equal(
        _frame_indices(20, start=2, stop=18, stride=3, maximum=4),
        [2, 5, 8, 11],
    )


def test_observation_indices_follow_checkpoint_timestamp_step() -> None:
    timestamps = np.array([1_000_000, 1_080_000, 1_160_000, 1_240_000])

    indices = _observation_image_indices(
        timestamps,
        np.array([0, 2, 3]),
        observation_steps=2,
        observation_step_s=0.1,
    )

    np.testing.assert_array_equal(indices, [[0, 0], [1, 2], [2, 3]])


def test_resolve_episode_from_runs_directory(tmp_path: Path) -> None:
    episode = tmp_path / "episode_0007_20260801_120000.h5"
    episode.touch()

    assert resolve_episode(tmp_path, 7) == episode.resolve()
    with pytest.raises(ValueError, match="--episode is required"):
        resolve_episode(tmp_path, None)


def test_visualization_input_contract_rejects_mismatched_streams() -> None:
    chunk = np.zeros((2, 3, 7), dtype=np.float64)
    frames = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="dp_action_chunk"):
        _validate_visualization_inputs(chunk, frames, sample_count=1)


def test_mujoco_visualization_draws_chunk_command_and_camera() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="gripper_base" pos="0 0 0.2">
              <geom type="sphere" size="0.02"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, 16)

    class _Viewer:
        user_scn = scene
        viewport = mujoco.MjrRect(0, 0, 800, 600)

        def __init__(self):
            self.images = []
            self.texts = []
            self.sync_count = 0

        def lock(self):
            return nullcontext()

        def set_images(self, value):
            self.images.append(value)

        def set_texts(self, value):
            self.texts.append(value)

        def sync(self):
            self.sync_count += 1

    viewer = _Viewer()
    chunk = np.array(
        [
            [0.1, 0.2, 0.3, 0, 0, 0, 1],
            [0.2, 0.3, 0.4, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    _sync_mujoco_action_visualization(
        SimpleNamespace(model=model),
        data,
        viewer,
        chunk,
        frame,
        frame_index=7,
        end_effector_body="gripper_base",
    )

    assert scene.ngeom == 3
    np.testing.assert_allclose(scene.geoms[0].pos, chunk[0, :3])
    np.testing.assert_allclose(scene.geoms[1].pos, chunk[1, :3])
    assert viewer.images
    viewport, image = viewer.images[-1]
    assert (viewport.width, viewport.height) == image.shape[1::-1]
    assert image.shape[-1] == 3
    assert viewer.texts and "frame 7" in viewer.texts[-1][2]
