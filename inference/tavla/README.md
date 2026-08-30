# ICRA2027 TA-VLA inference

This directory contains the TA-VLA deployment path migrated from
`tavla-client.zip`. The GPU machine hosts the native TA-VLA WebSocket policy;
the Nero computer captures two cameras and a 25 Hz torque history, requests
`[50, 7]` action chunks, and optionally executes the returned joint targets.

The three robot-side entry points have different safety scopes:

- `tavla_client.py`: saved-observation inference only; never opens the robot.
- `nero_tavla_runner.py`: one live inference; dry-run unless `--motion` is set.
- `nero_tavla_continuous.py`: asynchronous continuous inference; dry-run unless
  `--motion` is set.

## Start the policy server

From this Nero workspace on the TA-VLA GPU machine (the launcher changes into
`/data/jhr/TA-VLA` itself):

```bash
cd /path/to/Nero_ws
inference/tavla/run_inference_server.sh usb --gpu 0 --port 8000
```

The task can be `usb`, `button`, or `cucumber`. The launcher defaults to the
corresponding step-29999 checkpoint and refuses an incomplete checkpoint.

## Install the lightweight client

The robot computer needs this Nero workspace, OpenCV/pyAgxArm from its normal
environment, and `openpi-client` from the deployment bundle:

```bash
python -m pip install -e /path/to/tavla-client/code/inference/packages/openpi-client
python -m inference.tavla.tavla_client --self-test --task usb
```

To test the WebSocket server without opening cameras or CAN, use a saved
observation:

```bash
python -m inference.tavla.tavla_client \
  --host 192.168.100.101 \
  --port 8000 \
  --task usb \
  --camera-color bgr \
  --input sample.npz \
  --output runs/tavla/saved_observation_actions.npy \
  --max-joint-step 0.02
```

The NPZ must contain `side_image`, `wrist_image`, `state`, and
`effort_history`. The effort array can be the already sampled `[10, 7]` tensor
or at least 51 consecutive 25 Hz samples `[N, 7]`.

## One live observation

The defaults (`can1`, firmware profile `V112`, side camera 4, wrist camera 2)
match the supplied client archive. Override them when the robot computer uses
different stable device assignments.

First run inference without motion:

```bash
python -m inference.tavla.nero_tavla_runner \
  --host 192.168.100.101 \
  --task usb \
  --enable
```

This collects 2.2 seconds of torque, captures both cameras, saves the complete
action chunk to `runs/tavla/nero_tavla_actions.npy`, and prints the first
live-state-clipped target. It does not call `move_j`.

After inspecting that output and completing the robot-side safety checks, add
`--motion` to send exactly one target:

```bash
python -m inference.tavla.nero_tavla_runner \
  --host 192.168.100.101 \
  --task usb \
  --enable \
  --motion \
  --max-joint-step 0.02 \
  --speed-percent 5
```

## Continuous inference

The continuous runner samples torque at 25 Hz, consumes targets at 20 Hz, and
prefetches the next chunk when 20 actions remain. The full 50-step chunk is
kept; inference runs in a single worker so the timing loop does not block on
the WebSocket request.

Use a finite dry run first:

```bash
python -m inference.tavla.nero_tavla_continuous \
  --host 192.168.100.101 \
  --task usb \
  --max-actions 100
```

Add `--motion` only for an approved physical run:

```bash
python -m inference.tavla.nero_tavla_continuous \
  --host 192.168.100.101 \
  --task usb \
  --motion \
  --max-joint-step 0.02 \
  --action-rate 20 \
  --prefetch-actions 20
```

With motion enabled, non-NORMAL arm feedback or a disabled joint latches a
fault and clears the action queue. The latch never clears itself. Exiting does
not send `disable()` or an electronic emergency stop; the final issued target
is allowed to finish while the CAN connection remains open.

## Model contract

- side BGR camera -> RGB `images.cam_high`, resized/padded to `224x224`
- wrist BGR camera -> RGB `images.cam_left_wrist`, resized/padded to `224x224`
- current Nero J1-J7 position -> `state`, shape `[7]`
- J1-J7 torque at offsets `[-50,-44,-39,-33,-28,-22,-17,-11,-6,0]` from a
  25 Hz buffer -> `effort`, shape `[10, 7]`
- task language -> `prompt`
- server output -> first seven columns of `[50, >=7]`

All seven output dimensions are Nero joint targets. The upstream training
transform applies its delta/absolute conversion to the first six dimensions
and leaves the seventh absolute, but robot-side safety clipping applies to all
seven. The continuous runner clips twice: once against the observation state
after inference and again against the latest measured state immediately before
each possible command.

Keep the robot's validated joint limits, velocity limits, CAN watchdog,
workspace checks, and physical emergency stop active. The first inference can
be much slower because the server compiles JAX; warm it with motor output
disabled. The WebSocket is unauthenticated and unencrypted, so expose it only
on the trusted laboratory LAN.
