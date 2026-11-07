"""Standalone MuJoCo ball-kicker: a hinged arm striking a free ball.
Run: python labs/01_mujoco_kicker.py --headless
Requires: pip install mujoco numpy
This is a teaching surrogate, NOT the MicroDuck robot model.
"""
import argparse
import numpy as np
import mujoco

MODEL = r'''<mujoco model="kicker_tutorial">
<option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody>
  <geom name="floor" type="plane" size="2 2 0.1" rgba="0.7 0.7 0.7 1" friction="1 0.005 0.0001"/>
  <body name="base" pos="0 0 0.2">
    <geom type="box" size="0.08 0.08 0.2" mass="2" rgba="0.1 0.3 0.8 1"/>
    <body name="leg" pos="0.1 0 0">
      <joint name="hip" type="hinge" axis="0 1 0" range="-75 75" limited="true" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.26" size="0.035" mass="0.2" rgba="0.9 0.5 0.1 1"/>
    </body>
  </body>
  <body name="ball" pos="0.14 0 0.045">
    <freejoint/>
    <geom type="sphere" size="0.045" mass="0.05" rgba="0.9 0.1 0.1 1" friction="0.8 0.01 0.001"/>
  </body>
  <site name="target" pos="0.8 0 0.01" size="0.08" rgba="0 0.8 0 0.5"/>
</worldbody>
<actuator><motor joint="hip" gear="1" ctrlrange="-3 3" ctrllimited="true"/></actuator>
</mujoco>'''

def run(kick=2.4, headless=True):
    model = mujoco.MjModel.from_xml_string(MODEL)
    data = mujoco.MjData(model)
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
    if headless:
        for t in range(1500):
            data.ctrl[0] = kick if t < 125 else -0.8 if t < 260 else 0
            mujoco.mj_step(model, data)
        x, y, z = data.xpos[ball_id].copy()
        distance = np.hypot(x - 0.8, y)
        print(f'Ball at ({x:.3f}, {y:.3f}, {z:.3f}); target XY error={distance:.3f} m')
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            import time
            for t in range(1500):
                data.ctrl[0] = kick if t < 125 else -0.8 if t < 260 else 0
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--headless', action='store_true'); p.add_argument('--kick', type=float, default=2.4)
    a = p.parse_args(); run(a.kick, a.headless)
