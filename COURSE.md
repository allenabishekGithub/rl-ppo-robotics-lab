# Reinforcement Learning in MuJoCo
## From Simulation and PPO Training to ONNX and ROS 2 Policy Deployment

**Independent zero-to-advanced self-study course · MicroDuck-inspired capstone · November 2026**

**Learning objective:** Build a physics simulation; formulate an MDP; understand and train PPO; design and test a ball-kicking reward; evaluate policies; export a model to ONNX; understand how a real ROS 2 control bridge works; and investigate the official open-source MicroDuck training and runtime repositories.

> **Source and scope note.** This course is independently authored based on the workshop description provided by the learner and publicly available project documentation. It is **not** the official workshop, does not reproduce Bravo Robots' slides or materials, and does not claim that the exercise files are the official MicroDuck robot. The five lab scripts are intentionally smaller teaching systems. The real MicroDuck uses a different observation/action contract and its own trained policies.

## How to use this course

Allocate approximately **24–32 hours** over 8–12 sessions. Each section contains an explanation, equations and tasks. Run the included scripts for Labs 1–5, then complete the official MicroDuck track in a separate clone. A CUDA-capable GPU is needed for the standard mjlab/MuJoCo Warp GPU training pipeline. You can complete the core fundamentals on CPU. ROS 2 requires a separately sourced ROS 2 installation.

**Suggested deliverables:** a Git repository, screenshots of your MuJoCo scene, a reward comparison plot, a PPO training log, a passing PyTorch–ONNX equivalence test, a ROS 2 topic round-trip, a one-page sim-to-real safety checklist, and a concise technical report.

| Track | Tools | What works without an NVIDIA GPU? |
|---|---|---|
| Foundation | Python, NumPy, MuJoCo | Yes, small scenes |
| RL practice | Gymnasium, Stable-Baselines3 PPO | Yes, small tasks |
| Policy export | PyTorch, ONNX, ONNX Runtime | Yes |
| ROS 2 bridge | ROS 2 Humble, `rclpy` | Yes, on supported Ubuntu |
| Authentic MicroDuck training | `microduck_rl`, mjlab, MuJoCo Warp, PPO | Usually requires CUDA GPU or remote GPU |

---

# Part I — Robotics simulation foundations

## 1. What MuJoCo actually simulates

MuJoCo (Multi-Joint dynamics with Contact) numerically evolves a mechanical system subject to gravity, collisions, joint constraints, friction and applied forces. It does not decide *which* action a robot should take: the controller or learned policy supplies actions; MuJoCo computes the resulting state transition. A minimal robot has **bodies**, **joints**, **geometries**, **inertial properties**, **actuators**, **sensors** and an initial scene.

Let generalized coordinates be `q` (position/orientation/joint coordinates) and generalized velocities be `v`. A useful dynamics abstraction is

**M(q) · v_dot + C(q,v) + g(q) = τ + J(q)ᵀ f_contact.**

Here `M` is the mass matrix, `C` represents velocity-dependent terms, `g` gravity, `τ` actuator forces, and `Jᵀ f_contact` contact forces mapped into generalized coordinates. For a free-floating body, position/orientation use different coordinate parameterizations from velocities. A MuJoCo step integrates these dynamics under its contact solver and actuator model.

**Practical detail:** Simulation time step (`dt`) is not the same as policy update period. A physics engine may step at 2 ms while an RL policy acts every 20 ms. The same action is held across 10 physics steps (`decimation = 10`). Frequent policy actions increase control bandwidth and computing demands.

### MJCF anatomy

```xml
<mujoco model="demo">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="arm" pos="0 0 0.4">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.03 0.2" mass="0.3"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge" gear="1"/></actuator>
</mujoco>
```

A **geom** describes contact and/or visual shape; a **joint** defines permitted relative motion; an **actuator** maps commands to forces. Check units: SI metres, seconds and kilograms; angles and joint range conventions need care. Mass, moment of inertia, damping, coefficient of friction, motor limits and solver settings affect behavior.

### Lab 01 — Build a simple kicker

Install `requirements-core.txt` in a Python virtual environment. Run `python labs/01_mujoco_kicker.py --headless`, then on a graphical Ubuntu desktop run without `--headless`. Change `--kick 1.0` to `--kick 3.0` and compare the final distance from the ball to the target. The model is a **fixed-base hinged striker**, not a biped. Inspect `MODEL` in the source and identify every body, hinge, actuator, contact geom and site. Explain why a tiny ball is especially sensitive to initial collision geometry, friction and time stepping.

**Checks:** (1) the program prints an XY error; (2) changing command magnitude changes the trajectory in at least part of the explored range; (3) the state stays finite; (4) you can explain why the ball's `freejoint` adds translational and rotational degrees of freedom.

### Lab 02 — Physics sensitivity experiment

Change one variable at a time: `timestep` from 0.002 to 0.005, ball mass from 0.05 to 0.1 kg, friction from 0.8 to 0.4, and knee/hip damping. Run repeated trials and collect ball travel distance. Separate a physics parameter from a controller parameter; explain how contact parameters can affect policy transfer to real hardware. Record failures and numerical instabilities.

## 2. Robot control, coordinate frames and signals

Robot state combines joint angles `q`, velocities `q_dot`, orientation, angular velocity, and task geometry. Actions may be torque commands, desired positions, velocity targets, or normalized controls passed through an actuator mapping. A motor position controller often uses **τ = Kp(q_des − q) − Kd q_dot**, subject to saturation and motor dynamics. A learned action expressed as a position offset is not equivalent to a direct torque command.

Understand **world frame** versus **body frame**, rotations, quaternions, and end-effector position. For a planar two-link leg with link lengths `L1,L2` and joint angles `θ1,θ2`, the foot position is

**x = L1 cos θ1 + L2 cos(θ1 + θ2)** and **z = L1 sin θ1 + L2 sin(θ1 + θ2)**.

The Jacobian `J = ∂(x,z)/∂(θ1,θ2)` connects joint velocity to foot Cartesian velocity: `p_dot = J q_dot`. Close to straight-legged singularities, certain Cartesian directions become hard to realize with finite joint speeds.

### Lab 03 — Kinematics notebook

Write a NumPy function taking `θ1,θ2` to `(x,z)` and its analytical Jacobian. Approximate each Jacobian column with finite differences and assert the error is below `1e-4` for a non-singular configuration. Plot the reachable workspace. Discuss how impact point and foot velocity influence kick direction.

---

# Part II — Reinforcement learning from first principles

## 3. The Markov decision process

The control task is a Markov decision process `M = (S,A,P,r,γ)`. `S` is the state space, `A` the action space, `P(s'|s,a)` transition dynamics, `r(s,a,s')` the scalar reward and `γ` the discount. An observation `o_t` can be an incomplete representation of the physical state `s_t` (partial observability). A neural policy `πθ(a|o)` maps observations to an action distribution.

The expected discounted return at time `t` is **G_t = Σ(k=0…∞) γ^k r_(t+k)**. The objective is to maximize **J(θ) = Eπθ[G_0]**. The state value **Vπ(s) = Eπ[G_t | s_t=s]** and action value **Qπ(s,a) = Eπ[G_t | s_t=s,a_t=a]** satisfy Bellman relations, such as **Vπ(s) = E[r_t + γ Vπ(s_(t+1)) | s_t=s]**.

**Terminal vs timeout:** If the robot falls, the episode may truly terminate; if a maximum step count expires, it is often a time-limit truncation. Bootstrapping the value function across a timeout requires correct handling of truncated transitions. Gymnasium returns five values: `(observation, reward, terminated, truncated, info)`.

**Observation design:** A useful biped observation may include normalized joint positions/velocities, IMU readings, gravity projection, command vector, and previous action. Avoid using simulation-only information in an actor that will be deployed to hardware. Privileged state may be appropriate for a *critic* during training but not the onboard actor.

### Lab 04 — Hand-built MDP

For a 1-D ball task define state `(ball_x, ball_v)`, a discrete action `{-1,0,+1}` and dynamics `v_next = clip(v + 0.1a, -1,1)` and `x_next = x + 0.05v_next`. Use reward `-abs(target_x-x_next) - 0.02a²`, terminate on success and truncate at step 100. Simulate a random policy and a simple hand-coded policy; compare the mean return and success rate over 50 seeded episodes.

## 4. Policy gradients and actor–critic learning

Value-based methods such as DQN work naturally with discrete action sets. Continuous joint control is often better suited to policy-gradient and actor–critic approaches. For an on-policy policy-gradient algorithm, the actor parameter update is proportional to **E[∇θ log πθ(a_t|s_t) · A_t]** where `A_t` estimates how much better an action was than average.

With value baseline, a one-step advantage estimate is **δ_t = r_t + γ V(s_(t+1)) − V(s_t)**. Generalized Advantage Estimation (GAE) uses **Â_t = Σ(l=0…∞) (γλ)^l δ_(t+l)**, where `λ` tunes the variance/bias tradeoff. A critic is trained to predict returns, frequently with a squared loss.

What does actor–critic solve? The **actor** chooses an action and the **critic** estimates state quality. The baseline reduces variance in the gradient without changing its expected direction under standard assumptions. The learner still needs adequate exploration and stable reward scaling.

## 5. PPO — the central algorithm

Proximal Policy Optimization compares the new action probability with the rollout policy that generated the sample. The importance ratio is **ρ_t(θ) = πθ(a_t|s_t) / πθ_old(a_t|s_t)**. PPO's clipped surrogate is

**L_clip(θ) = E[min(ρ_t Â_t, clip(ρ_t,1−ε,1+ε) Â_t)]**.

For positive advantage, clipping avoids rewarding an excessive probability increase. For negative advantage, clipping bounds certain overly large decreases. This is a practical *surrogate objective*, not a hard mathematical guarantee that every update remains in a trust region. Typical PPO training combines this actor loss with value-function loss and an entropy bonus, e.g. **L_total = −L_clip + c_v L_value − c_e H(π)**.

### Step-by-step PPO iteration

1. Reset many environments with varied initial conditions and target commands.
2. Roll out `n_steps` using the current stochastic actor; store `(o,a,reward,done,log_prob,value)`.
3. Compute discounted returns and GAE advantages, respecting episode boundaries.
4. Normalize advantages when appropriate, then shuffle samples into minibatches.
5. Perform several epochs updating actor and critic using clipped objective.
6. Monitor policy entropy, value loss, approximate KL divergence and success rate.
7. Evaluate in held-out configurations, not just the training random seeds.

**Useful training knobs:** `γ=0.99`, `λ=0.95`, `clip_range=0.2`, Adam learning rate around `3e−4` are *starting points*, not universally optimal settings. Batch size, rollout length, action distribution standard deviation, normalization and reward scale can matter as much as nominal learning rate. Real robot training may need action decimation, actuator delay modeling and constrained controls.

### Lab 05 — PPO on Pendulum, CPU

Run `python labs/03_ppo_pendulum.py`. It trains a continuous-action PPO agent using Stable-Baselines3 and saves `ppo_pendulum.zip`. Watch actor/critic metrics in the console. Repeat with `clip_range=0.1` and `0.3`, and `learning_rate=1e-4` and `1e-3`. Use at least three random seeds for comparisons; avoid treating one run as evidence. Compute mean and standard deviation of evaluation return across 20 episodes.

### Lab 06 — Implement the PPO loss yourself

Starting from NumPy arrays of sample `old_logp`, `new_logp`, and `advantage`, calculate `ratio = exp(new_logp − old_logp)` and `min(ratio*A, clip(ratio,1−eps,1+eps)*A)`. Test the four cases: positive/negative advantage and ratios above/below clipping limits. Explain why `min` acts differently when the advantage is negative. Compare your outputs with an equivalent PyTorch tensor implementation.

---

# Part III — Reward engineering for the ball-kicking task

## 6. From hopscotch requirements to measurable objectives

The user goal is **kick a ball with the correct force so it stops inside a chosen square**, while avoiding falls and excessive energy use. Convert this into observable quantities. A target square can be defined by center `(x*,y*)`, half-width `h`, and a ball stopping condition `||v_ball|| < v_stop`. A binary success is true only if the ball is in the square **and** nearly stationary. A ball passing through the square at high speed must not earn full success credit.

**Sparse success:** `R_success = 1` on satisfying the target condition and `0` otherwise. This directly matches the goal but may be difficult to learn. **Shaped reward:** `R_dist = exp(−k||p_ball−p_target||²)` gives dense feedback but risks encouraging an agent to leave the ball near the target or exploit the simulator without truly kicking.

One illustrative episodic reward is

**r_t = w_g·1_success + w_p·exp(−k d_ball,target²) − w_f·1_fall − w_e·||τ||² − w_s·||a_t−a_(t−1)||²**.

These terms have different roles: goal achievement, dense progress, stability, energy and smooth control. They must be scaled and audited. Another option is potential-based shaping **F(s,s')=γ Φ(s')−Φ(s)**; with standard assumptions, this preserves an optimal policy more reliably than arbitrary shaping.

### Critical questions before coding

- Is the target ball center, landing point, or final resting point?
- At what moment is the ball's stopping position measured?
- Is the robot permitted to step forward? Is the kick single-shot or multi-step?
- Are contacts with hands, body or floor allowed? What counts as a foul?
- Should one policy handle many square locations, or should each square use a separate policy?
- Which variables exist on physical hardware and which are simulator-only?

## 7. Trainable environment specification

**Suggested observation** (teaching environment): normalized joint angles, joint velocities, body tilt, angular velocity, previous action, relative ball position, relative target position, ball velocity and a phase flag. **Suggested action:** 14 normalized joint targets for authentic MicroDuck-like control, or one actuator command in the provided teaching surrogate. Crucially, the actual official MicroDuck uses a **61-dimensional actor observation and 14-action policy** and has a defined order for input features. Do not assume the teaching observation matches that contract.

**Reset distribution:** Randomize target center, ball pose, robot initial posture and perhaps physical dynamics. **Termination:** fall, unsafe contact, achieved success, joint limit violation or simulation numerical error. **Truncation:** episode time budget. **Success metric:** fraction of held-out trials where ball is stationary in square and robot remains upright.

### Lab 07 — Reward sweep, no training required

Run `python labs/02_reward_sweep.py`. The script varies one motor command, measures final XY target error, and compares a shaped reward with an energy penalty. It plots `reward_sweep.png`. Identify the optimal action under each reward. Replace the penalty coefficient 0.01 with 0.0 and 0.1: how does the best action change? Extend to 50 trials with randomized ball mass and report mean and worst-case target error.

### Lab 08 — Build a Gymnasium wrapper

Create `KickerEnv(gym.Env)` from Lab 01. Implement `observation_space`, `action_space`, `reset(seed, options)`, `step(action)` and `render()`. Start with an action held for 125 physics steps, then choose to (a) run an entire kick per RL episode, or (b) act every 10 physics ticks with a 300-step horizon. Do **not** mix these without documenting the control frequency. Make observations finite float32 arrays; report target error and success in `info`.

### Lab 09 — Train PPO on the new wrapper

Replace `Pendulum-v1` in Lab 05 with your kicker environment. Begin with a fixed target, then use random targets. Train 3 seeds and compare results against random search of one-shot kick strength. If the one-joint PPO agent does not beat a simple optimizer, investigate contact geometry, reward identifiability, stochasticity, normalization and action duration **before** scaling to a full robot.

### Lab 10 — Reward ablation

Train or evaluate the same environment with: (A) target-only reward, (B) target + progress, (C) target + progress + energy, (D) all terms. Log final distance, success, falls and energy per episode, not merely total reward. A higher training reward with a lower success rate is a warning sign for reward hacking. Save videos of representative failures.

---

# Part IV — Authentic MicroDuck and mjlab

## 8. The open-source codebases

The official, publicly documented MicroDuck project provides a learning reference:

- **RL training:** https://github.com/pollen-robotics/microduck_rl
- **Robot/runtime + simulated duck:** https://github.com/pollen-robotics/microduck
- **mjlab:** https://github.com/mujocolab/mjlab
- **ROS 2 Humble:** https://docs.ros.org/en/humble/

According to its current repository documentation, MicroDuck is a small bipedal robot. Training uses mjlab (MuJoCo Warp) and PPO. The exported ONNX model is run by the separate robot runtime. The repository contains a ball-kick task family, observation/reward logic, training scripts and export scripts. **Repository CLI interfaces evolve; verify the commands against the checked-out README before executing.**

### Understand the core source layout

`src/mjlab_microduck/robot/` contains robot assets and MJCF; `tasks/` contains task definitions and environment configurations; `tasks/mdp.py` contains reward and observation building blocks; the actuator layer models motor physics and randomization; `scripts/export.py` handles compliant ONNX export. The current public documentation specifies **61 inputs and 14 outputs** for deployable MicroDuck actor policies, with observation normalization baked into the official export. A hand-converted checkpoint may be unsafe or incompatible.

**Architectural insight:** A policy that works in a fast idealized physics simulator may fail on real actuators because of friction, battery voltage, input delay, backlash, quantization, contact variability and sensing differences. The MicroDuck training implementation addresses these explicitly.

### Lab 11 — Reproduce the repository environment

```bash
git clone https://github.com/pollen-robotics/microduck_rl.git
cd microduck_rl
# Inspect README.md, pyproject.toml, scripts/ and task registration first.
# Follow the repository's current instructions for installing uv and dependencies.
```

Inspect the actual task names in the code. Locate the BallKick environment configuration, its observations, reward components, termination conditions, actuator limits and randomization toggles. Draw a table of observation name, dimension, normalization and physical availability. Compare that table to your teaching environment.

### Lab 12 — CPU-only replay / GPU training

If the upstream project and dependencies support your operating system and GPU, follow the **current** repository quickstart for training a simple policy, then evaluate it in simulation. A documented example task is `Mjlab-Velocity-Flat-MicroDuck`, but check task registration before running. If no CUDA GPU is available, do not force GPU training on a CPU-only installation: first use official prerecorded policies and the CPU inference/viewer path described in the project, or a remote CUDA instance.

**Experiment:** Run a pretrained policy twice with changed floor friction or a changed commanded velocity. Observe whether joint excursions, stability and target tracking degrade. Save a short screen recording of one nominal and one perturbed rollout.

### Lab 13 — Ball-kick task inspection

Identify where the ball's final position and velocity are measured. Find how the kicking leg or phase is commanded, whether the task contains a fixed target or a range of target points, and whether success evaluation happens after the ball stops. Write a one-page proposed modification for hopscotch squares. **Do not assume that the repository's existing ball-kick task already implements the workshop's exact game.**

### Lab 14 — Train two reward variants in official environment

When a CUDA GPU is available, establish a baseline task and record its configuration and commit hash. Train a variant with modified target tolerance and a second with a larger energy penalty. Keep equal training budget and seed sets. Export an evaluation table with success rate, landing error, fall rate, and wall-clock inference time. Inspect random videos rather than only the highest-score video.

---

# Part V — From checkpoint to production inference

## 9. Why ONNX?

Training typically produces a PyTorch checkpoint containing neural weights and optimizer metadata. ONNX stores a portable computational graph for efficient runtime inference, often without the Python training framework. **Export correctness has two parts**: numerical parity between PyTorch and ONNX Runtime, and semantic parity of the full pipeline (observation ordering/scaling, action interpretation, stateful history, clipping and control interval).

For observation `o` and actor `fθ`, deployment computes `a = fθ(N(o))`, where `N` is the training-time normalization transform. Omitting `N` can make a perfectly exported neural network behave incorrectly. Check input/output shapes, dtype, device, latency, NaN handling and model version.

### Lab 15 — Export the teaching PPO actor

After Lab 05, run `python labs/04_export_onnx.py`. The example exports the **deterministic Pendulum PPO actor** (3 input values, 1 output) and checks PyTorch–ONNX inference agreement. This is **not** a MicroDuck policy. Test 20 randomly generated inputs and assert maximum absolute difference under a justified tolerance; record how export dependencies and opsets can affect compatibility.

### Lab 16 — Verify official MicroDuck policy contract

Use **the official** `scripts/export.py` with its documented flags to export a policy from a supported training run. Validate `[1,61] -> [1,14]`, dtype, finite outputs, normalizer inclusion, and timing under representative input distributions. Reject mismatched models before ROS 2 or robot deployment. A successful `onnx.checker.check_model` verifies graph structure, not correct control behavior.

## 10. ROS 2 concepts and robot control architecture

ROS 2 applications are composed of **nodes** that publish and subscribe to **topics**, offer **services**, and execute **actions**. A topic suits continuous state data and commands; an action suits a long-running high-level goal with feedback; a service suits short request–response calls. Nodes may use lifecycle transitions to prevent issuing commands before sensors, model and actuators are ready.

Proposed architecture:

```text
MuJoCo physics / hardware drivers
  | joint states, IMU, ball pose, target command
  v
Observation builder (ordering, normalization, freshness)
  | float32 policy observation
  v
ONNX policy node (fixed rate, input checks)
  | normalized action
  v
Action adapter (scale, clip, watchdog, interlocks)
  | actuator setpoints
  v
MuJoCo or hardware controller
```

A 50 Hz policy loop has a **20 ms** cycle budget, including sensing, preprocessing, inference, communication and command output. This is a time-budget example rather than a promise about specific hardware latency. Every observation needs a timestamp; drop stale data; hold or safely stop on missed deadlines. For physical robots, add an independent emergency stop and explicit limits, not just a neural policy's learned preferences.

### Lab 17 — ROS 2 node: messages and callback

On Ubuntu 22.04 with ROS 2 Humble installed and sourced, launch `python labs/05_ros2_onnx_node.py` after Lab 15 in the directory containing `pendulum_actor.onnx`. In another sourced terminal:

```bash
ros2 topic echo /rl_action
ros2 topic pub --once /rl_observation std_msgs/msg/Float32MultiArray \
  "{data: [1.0, 0.0, 0.0]}"
```

The node should publish one action. Test malformed observations of length two and NaNs; the minimal node rejects bad dimensionality and non-finite outputs, but production code should validate *inputs* more strictly as well. For the actual robot, replace these temporary message contracts with the appropriate observation builder and action interface. **Never plug this tutorial's 3→1 Pendulum policy into MicroDuck hardware.**

### Lab 18 — ROS 2 package, launch files and watchdog

Make a Python `ament_python` package containing the policy node. Add parameters `model_path`, `control_frequency`, and `max_observation_age_ms`; a launch file; startup checks; and a watchdog that publishes safe neutral actions when state updates stop. Record the actual node startup and shutdown sequence. Test with intentionally delayed observations, incorrect model dimensions and artificial inference lag.

### Lab 19 — End-to-end simulation

Connect MuJoCo state to ROS 2 via a bridge that periodically publishes observations and subscribes to actions. Run the policy node as an independent process. Log wall-clock latency distribution (p50/p95/p99), missed deadlines and command saturation, then compare trajectories to direct in-process inference. A simulated trajectory matching direct inference is a basic integration check; it is **not** sufficient evidence of physical safety.

---

# Part VI — Robustness, safety, evaluation and architecture

## 11. Sim-to-real gap and domain randomization

Define uncertain parameters `ξ = {mass, friction, motor gain, battery voltage, delays, sensor noise, contact restitution}`. Instead of training only at a nominal `ξ0`, draw `ξ ~ p(ξ)` at each reset. The robust objective is **maxθ Eξ~p(ξ)[J(πθ; ξ)]**, though the selected distribution can bias performance and no finite randomization guarantees safety outside its range.

Test both nominal and out-of-distribution scenarios. Useful perturbations include: low friction, low battery voltage, robot yaw offset, shifted ball starting position, unseen target squares, perturbation midway through a kick and delayed observations. Distinguish **training randomization**, **validation randomization** and **final challenge cases**.

### Lab 20 — Domain randomization stress test

Train or evaluate at mass multipliers `0.8–1.2`, friction `0.5–1.2`, action delay `0–2` policy cycles and target offsets. Report success as a matrix rather than a single average; identify catastrophic corners. Plot landing-error histograms and compare the median with 95th percentile.

## 12. Verification, reproducibility and safety

Version your code, task environment configuration, simulator version, seeds, model artifact, normalizer, observation schema and reward weights. A deployment bundle should include a manifest with input/output dimensions, joint mapping, expected control frequency, action scale, validation result and policy checksum. Log training with TensorBoard, MLflow or Weights & Biases, but do not confuse monitoring integration with algorithmic correctness.

**Safety gates before real hardware:** (1) explicit joint/action limits and rate limits; (2) watchdog and emergency stop; (3) stale-observation detection; (4) verified policy tensor contract; (5) offline simulations covering falls, collisions and contact extremes; (6) supervised low-power test in a controlled physical area. **This course does not authorize unattended physical deployment.**

### Lab 21 — Evaluation harness

Run each candidate on at least 100 held-out seeded episodes spanning target squares and physics perturbations. Compute **success rate** `sum(success_i)/N`, mean/median/p95 final positional error, fall rate, energy, and inference deadline misses. For a binomial success rate `p̂`, an approximate standard error is `sqrt(p̂(1−p̂)/N)` away from pathological boundaries; consider Wilson intervals for small N or near-0/1 rates. Compare policies on matched scenarios.

### Lab 22 — Failure analysis

Tag failure videos as (a) wrong contact timing, (b) excessive impulse, (c) weak kick, (d) falls, (e) sensor/normalizer mismatch, (f) out-of-distribution friction, (g) late control. Provide an engineering hypothesis and next experiment for each. A good research report discusses failures rather than hiding them.

## 13. Optional extension — agentic task orchestration

A higher-level agent may choose target square, invoke perception, schedule an already validated skill and evaluate success. It should **not** emit unbounded motor torques or directly modify low-level control limits. Keep the low-level PPO loop deterministic and time-bounded, and the agentic planner asynchronous. This separates deliberative decision-making from real-time closed-loop control.

Proposed orchestration example: `task.request(target_square) -> validate_target -> select_policy('kick') -> configure_command -> start_episode -> observe_result -> report_success/failure`. Add a supervisor node that can cancel the skill. This extension connects robotics to orchestration systems without treating LLM output as safety-critical actuator control.

### Lab 23 — Skill executor (optional)

Design a ROS 2 action `KickToSquare` (goal: square ID/pose; feedback: phase and estimated confidence; result: success and final ball position). Build a finite-state machine `IDLE -> PREPARE -> EXECUTE -> VERIFY -> DONE/ABORT`. Instrument cancellations, timeouts and duplicate requests. Evaluate the orchestration independently of the low-level policy.

---

# Part VII — Capstone and assessment

## 14. Final capstone: MicroDuck hopscotch

**Goal:** In simulation, a biped robot receives a chosen square; strikes a ball; and achieves high landing accuracy while staying upright. A full attempt should include an authentic robot model and task, trained PPO policy, realistic evaluation scenarios, correctly exported ONNX model and a simulation-only ROS 2 policy runner. The starter scripts teach pieces of this pipeline; they do not constitute a finished capstone implementation.

### Phase A — Baseline (evidence required)

Clone and inspect MicroDuck source; document commit and versions; establish a supported simulation environment; replay a pretrained baseline; save an annotated screenshot. Identify its observation schema, action mapping and model normalization.

### Phase B — Task implementation

Define configurable square targets; observation construction; allowed contact semantics; episode termination; accurate target-stop success conditions; curriculum and reward components. Add unit tests for reward signs, reward monotonicity for distance, stable resets, NaN guards and ball-target geometry.

### Phase C — Training and experiments

Run PPO training with at least three seeds when computationally feasible. Compare baseline reward against two improved designs. Archive run configuration and top checkpoints by **held-out success**, not by training reward alone. Record learning curves and representative failure videos.

### Phase D — Runtime deployment in simulation

Use the upstream exporter (not ad-hoc conversion) and validate official tensor contract. Run CPU policy replay first; integrate ROS 2 via explicit observation and action adapters; test deadlines, stale-data handling, joint clamping and simulated shutdown behavior. Physical deployment is an optional separate, supervised engineering project.

### Phase E — Scientific reporting

Include problem statement, dynamics assumptions, observation/action specification, PPO configuration, reward equations, randomization ranges, training curves, held-out results, ablation table, failure cases, resource usage, and limitations. Distinguish results achieved in the teaching surrogate from those achieved using the full MicroDuck stack.

## 15. Assessment rubric

| Criterion | Weight | Evidence |
|---|---:|---|
| Physics and modelling | 15% | Model, actuator, joints, contact tests |
| RL/PPO understanding | 20% | Derivation, algorithm explanations, stable learning |
| Reward and task engineering | 20% | Reward audit, ablations, success definition |
| Evaluation rigor | 20% | Held-out trials, robustness, repeatability |
| ONNX/ROS 2 integration | 15% | Parity tests, policy bridge, timing and safety |
| Documentation | 10% | Reproducibility, limitations, repository |

**Passing target:** 70/100 with no unaddressed critical safety or model-contract failures. **Advanced target:** 90/100 with reproducible multi-seed results, randomized test scenarios and transparent failure analysis. Numerical ball-landing performance thresholds must be established after observing a realistic baseline; this document does not promise any success rate.

## 16. Viva / interview questions with short answers

**Q1. Why MuJoCo?** Fast rigid-body simulation with contacts and articulated joints; used to evaluate control policies under physics.

**Q2. Why PPO instead of DQN?** PPO natively supports continuous actions; DQN's basic variant assumes discrete actions.

**Q3. What is the advantage function?** `A(s,a)=Q(s,a)-V(s)`; it estimates the relative benefit of an action.

**Q4. What does PPO clip?** The policy-probability ratio in the surrogate objective, not the physical motor action.

**Q5. Why normalize observations?** To stabilize learning and inference; deployment must reproduce the same transform.

**Q6. What is reward hacking?** Exploiting a proxy reward while failing the actual task objective.

**Q7. What is domain randomization?** Training across sampled dynamics/sensing variations to improve robustness.

**Q8. How is a checkpoint different from ONNX?** A training checkpoint includes state and optimizer context; ONNX describes a portable inference graph.

**Q9. Why ROS 2?** Interoperable runtime messaging, lifecycle management and modular hardware/simulation integration.

**Q10. Why is a pretrained policy not automatically safe?** Distribution shift, actuator mismatch, input-schema errors, latency and untested failures.

**Q11. How do you evaluate kick quality?** Held-out success with stationary ball inside target, positional error, robot falls and effort.

**Q12. What is a policy frequency?** Rate at which neural actions are generated, distinct from physics integration frequency.

---

# Appendix A — 10-day suggested schedule

| Day | Study | Evidence |
|---|---|---|
| 1 | MuJoCo, MJCF, physics | Lab 01 screenshot and observations |
| 2 | Kinematics and physics sensitivity | Labs 02–03 report |
| 3 | MDP, values, returns | Lab 04 notebook |
| 4 | Actor–critic, GAE and PPO | Lab 06 math checks |
| 5 | PPO CPU implementation | Lab 05 checkpoint and metrics |
| 6 | Target/reward engineering | Labs 07–08 plots and environment |
| 7 | PPO kicker and reward ablation | Labs 09–10 experiment table |
| 8 | MicroDuck source and tasks | Labs 11–13 architecture notes |
| 9 | ONNX and ROS 2 | Labs 15–18 demos |
| 10 | Robustness and capstone | Labs 20–23 roadmap, final report |

The full authentic GPU training and repeated capstone experimentation may take additional days. The schedule is a learning structure, not a guarantee of completion time.

# Appendix B — Troubleshooting

**MuJoCo viewer unavailable:** Headless mode works without desktop/OpenGL display. For interactive rendering, use a local graphical session and appropriate GPU/display drivers. macOS may need the designated launcher for MuJoCo viewer.

**Model fails to compile:** Check XML validity, masses/inertias, joints, actuator names, geom dimensions, initial interpenetration and version-specific MJCF attributes.

**PPO fails to learn:** Verify that the environment responds to actions; rewards reflect objective; actions are scaled to actuator ranges; observations are finite; episodes reset; time limits are correctly handled; success signals are not vanishingly rare.

**ONNX outputs differ:** Check deterministic versus stochastic actor, model in evaluation mode, input float32 shape, normalization, exporter opset and compatible ONNX Runtime.

**ROS 2 topics show no data:** Confirm `source /opt/ros/humble/setup.bash`, nodes running, topic names/types and compatible `ROS_DOMAIN_ID`. Check QoS if discovery succeeds but messages do not arrive.

**Simulation policy works but robot fails:** Validate observation ordering, frame conventions, joint indexing, motor command units, actuator physics, latency, battery behavior and friction. Stop the hardware test if any mismatch occurs.

# Appendix C — Verified starting resources (accessed November 2026)

- Pollen Robotics, *microduck_rl*, repository and training documentation: https://github.com/pollen-robotics/microduck_rl
- Pollen Robotics, *microduck* runtime, simulated duck and policy infrastructure: https://github.com/pollen-robotics/microduck
- mjlab official code: https://github.com/mujocolab/mjlab
- MuJoCo documentation: https://mujoco.readthedocs.io/
- ROS 2 Humble docs: https://docs.ros.org/en/humble/
- Gymnasium API: https://gymnasium.farama.org/
- Stable-Baselines3 PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- ONNX: https://onnx.ai/ and ONNX Runtime: https://onnxruntime.ai/

**Final reminder:** Your supplied workshop advertises MicroDuck and ROS 2. The public Pollen Robotics runtime is a distinct Rust-based architecture; this course's ROS 2 bridge is an independent reproduction of the *learning objective*, not a claim that the public MicroDuck production runtime is implemented in ROS 2.
