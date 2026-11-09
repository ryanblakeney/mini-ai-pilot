# mini-ai-pilot

Notes, tools and plans for letting a trained reinforcement-learning pilot fly a real DJI Mavic Mini (Mini 1) through its own remote controller, with on-phone computer vision (follow a target, avoid obstacles). Work in progress, written as a build journal: what worked, what did not.

## What works today (verified)
- A Raspberry Pi Zero 2 W can act as the "phone" for the Mini's remote (USB gadget mode), using the open-source DJI Link bridge, so the remote opens its full link. A PC plugged straight into the remote cannot reach the aircraft.
- Aircraft telemetry streams to the computer through that path, and the gimbal responds to a command (`tools/mavic_link.py`).
- In simulation (Project AirSim), a Soft Actor-Critic policy holds a hover in gusty wind with 0.335 m mean error (zero-action 1.87 m, hand-written controller 0.41 m).

## What is NOT proven yet
Flight commands (virtual stick) on the real aircraft, live video decoding, on-phone model speed, and sim-to-real transfer. Each has a planned test in `docs/`.

## Contents
- `docs/REAL_AIRCRAFT_LINK.md`: how the link works, message format, gimbal command, safety rules, the plan for flying.
- `docs/VISION_ONBOARD_RESEARCH.md`: where perception and the pilot run (phone vs laptop), models, licenses, test ladder.
- `docs/VISION_AVOIDANCE_RESEARCH.md`: camera-based obstacle avoidance research.
- `sim/`: the AirSim gym environment, SAC training script, and the waypoint (stage 2) environment with its reward check. Paths are from the original Windows setup (`D:\fpv\...`); adjust them.
- `ccw/`: a reduced-scale implementation of concept-based explanations (CCW-Net, arXiv 2603.21369) applied to the hover policy. Experimental and not yet validated against the paper.
- `tools/mavic_link.py`: read gimbal pitch, move the gimbal one safe step.

## Safety
Never fly near people, roads or restricted airspace; check local rules; remove propellers for every bench test. The code here sends no flight commands. Use at your own risk. The DUML protocol notes come from community reverse engineering and may be wrong or change with firmware.

## License
Apache-2.0 (see LICENSE). Third-party projects keep their own licenses.
