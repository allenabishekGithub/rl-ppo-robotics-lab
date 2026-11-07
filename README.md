# MuJoCo -> PPO -> ONNX -> ROS 2: Independent Course Labs

These are **educational surrogate experiments**, not the official Bravo Robots workshop materials and not a deployable MicroDuck policy.

## Setup
On Ubuntu 22.04/24.04 with Python 3.10+:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
```

## Sequence
```bash
python labs/01_mujoco_kicker.py --headless
python labs/02_reward_sweep.py
python labs/03_ppo_pendulum.py
python labs/04_export_onnx.py
```
For GUI mode on Linux, run `python labs/01_mujoco_kicker.py` in a graphical desktop session. Lab 05 requires a separately installed and sourced ROS 2 environment, and launches with `python labs/05_ros2_onnx_node.py` from the folder containing `pendulum_actor.onnx`. Publish: `ros2 topic pub --once /rl_observation std_msgs/msg/Float32MultiArray "{data: [1.0, 0.0, 0.0]}"`; echo `/rl_action`.

## Official MicroDuck stack
Use https://github.com/pollen-robotics/microduck_rl for the *actual* 61-observation, 14-action environment and its own export script. Do not load the example Pendulum model on MicroDuck. Follow the repository's **current README**, as dependencies and CLI flags change. Real policy execution requires safety engineering, rate/latency tests, action constraints, and supported hardware.
