from types import SimpleNamespace
import numpy as np
import pytest
from inference.pi0_wm.core import Plans, MissingActions, Result, Request, aligned_pi_schedule
from inference.pi0_wm.runtime import Runtime


def camera_runtime(monkeypatch):
    import inference.pi0_wm.runtime as module
    monkeypatch.setattr(module.time, 'time', lambda: 1.1)
    rt = Runtime.__new__(Runtime)
    rt.cfg = {'pi0': {'prompt': 'test', 'interface': {
        'images': {'side': 'side', 'wrist': 'wrist'},
        'state_key': 'state', 'prompt_key': 'prompt'}},
        'control': {'maximum_camera_age_s': .3}}
    rt.frames = {name: SimpleNamespace(timestamp_us=stamp,
                 frame=np.zeros((2,2,3),np.uint8))
                 for name, stamp in [('side', 1010000), ('wrist', 1015000)]}
    rt.pi_clock_samples = []
    for i in range(11):
        pose = np.eye(4); pose[0,3] = i
        rt.pi_clock_samples.append((1+i*.01, 100+i, SimpleNamespace(ee_pose=pose)))
    rt.step = 110
    rt.state = SimpleNamespace(q=np.zeros(7))
    rt.wm_enabled = True
    rt.plans = Plans(20)
    rt.request_id = 0
    rt.pi_worker = SimpleNamespace(submit=lambda request: None)
    return rt


def test_pi_request_uses_wrist_time_and_causal_state(monkeypatch):
    rt = camera_runtime(monkeypatch)
    request = rt.submit_pi()
    assert request.anchor == pytest.approx(101.5)
    assert request.payload['state'][0] == 1  # State at 1.010, not latest at 1.100.
    assert rt.pi_snapshot_age_s == pytest.approx(.085)
    plan = rt.plans.add(actions(0), 28, request.request_id, request.anchor)
    assert plan.action_at(28,4)[0] == 2
    assert plan.length == 20


def test_camera_anchor_uses_observed_clock_under_loop_jitter(monkeypatch):
    rt = camera_runtime(monkeypatch)
    rt.pi_clock_samples = [(1., 100, rt.pi_clock_samples[0][2]),
                           (1.06, 101, rt.pi_clock_samples[1][2])]
    assert rt.submit_pi().anchor == pytest.approx(100.25)


@pytest.mark.parametrize('stamp', [999000, 1101000, 0])
def test_unbracketed_future_or_stale_camera_waits(monkeypatch, stamp):
    rt = camera_runtime(monkeypatch)
    rt.frames['wrist'].timestamp_us = stamp
    with pytest.raises(MissingActions):
        rt.submit_pi()


def actions(anchor, length=50):
    values = np.zeros((length, 7), dtype=np.float32)
    values[:, 0] = anchor / 4 + np.arange(length)
    values[:, 6] = 1
    return values


def test_logged_800ms_gap_skips_20_actions_and_retains_only_valid_suffix():
    plans = Plans(50)
    plan = plans.add(actions(640), 180, 1, 640)
    assert plan.length == 30
    assert plan.action_at(180, 4)[0] == 180
    assert plan.action_at(209, 4)[0] == 209
    with pytest.raises(MissingActions):
        plans.window(209 * 4, 2, 0)


def test_cross_boundary_window_is_time_aligned_even_for_early_observation():
    plans = Plans(30)
    plans.add(actions(0), 0, 0, 0)
    plans.add(actions(40), 30, 1, 40)
    values, versions = plans.window(25 * 4, 10, 1)
    np.testing.assert_array_equal(values[:, 0], np.arange(26, 36))
    assert versions == (0, 1)


def test_fractional_phase_uses_previous_source_action_without_reanchoring():
    plans = Plans(20)
    plan = plans.add(actions(3), 5, 0, 3)
    assert plan.action_at(5, 4)[0] == 4.75
    assert plan.action_at(6, 4)[0] == 5.75


def test_expired_chunk_is_not_restarted_or_padded():
    plans = Plans(20)
    with pytest.raises(MissingActions, match='expired'):
        plans.add(actions(0), 50, 0, 0)
    assert not plans.plans


@pytest.mark.parametrize('horizon', [10, 20])
def test_aligned_schedule_runs_many_chunks_without_window_gaps(horizon):
    rt = Runtime.__new__(Runtime)
    consume, lookahead = aligned_pi_schedule(50, 50, horizon, 1, .25, .08, 25)
    rt.plans = Plans(consume)
    rt.pi_lookahead = lookahead
    rt.pi_requested_tail = None
    rt.pi_handoff = 0
    rt.last_plan = None
    rt.log_pi_chunk = lambda: None
    rt.step = 0
    jobs = []
    submitted = []
    worker = SimpleNamespace(busy=False)
    def submit():
        submitted.append((rt.step, bool(rt.plans.plans) and rt.step < rt.plans.plans[-1].start_token * 4))
        request = Request(len(submitted), rt.step, (), None, 0)
        jobs.append((rt.step + 24, Result(request, actions(rt.step), .24)))
        worker.busy = True
    def poll():
        if jobs and jobs[0][0] <= rt.step:
            worker.busy = False
            return jobs.pop(0)[1]
    worker.poll = poll
    rt.pi_worker = worker
    rt.submit_pi = submit
    ready = False
    for step in range(800):
        rt.step = step
        rt.pi_update()
        try:
            values, _ = rt.plans.window(step, horizon, 1)
        except MissingActions:
            assert not ready, f'future coverage lost at {step}'
        else:
            ready = True
            np.testing.assert_array_equal(values[:, 0], np.arange(step // 4 + 1, step // 4 + 1 + horizon))
    assert ready and len(submitted) > 5
    if horizon == 20:
        assert any(pending for _, pending in submitted[1:])


def test_infeasible_time_aligned_schedule_is_rejected():
    with pytest.raises(ValueError, match='too short'):
        aligned_pi_schedule(50, 50, 20, 1, .6, .08, 25)


def test_late_result_keeps_original_time_axis_and_only_remaining_tail():
    plans = Plans(30)
    plans.add(actions(0), 0, 0, 0)
    # Handoff was token 30, but the result arrives after it at token 43.
    late = plans.add(actions(40), 43, 1, 40)
    assert late.length == 17
    assert late.end_token == 60
    np.testing.assert_array_equal(plans.window(43 * 4, 10, 0)[0][:, 0], np.arange(43, 53))
    with pytest.raises(MissingActions):
        plans.window(60 * 4, 1, 0)


def test_runtime_discards_fully_expired_response_and_requests_fresh_observation():
    rt = Runtime.__new__(Runtime)
    rt.step = 240
    rt.plans = Plans(20)
    rt.pi_handoff = 0
    rt.pi_lookahead = 20
    rt.pi_requested_tail = None
    rt.log_pi_chunk = lambda: None
    rt.pi_worker = SimpleNamespace(busy=False, poll=lambda: Result(Request(1, 0, (), None, 0), actions(0), 2.4))
    requests = []
    rt.submit_pi = lambda: requests.append(rt.step)
    rt.pi_update()
    assert not rt.plans.plans
    assert requests == [240]
