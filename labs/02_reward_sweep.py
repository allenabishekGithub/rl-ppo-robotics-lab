"""Compare a reward and success metric for different kick strengths.
Requires: mujoco numpy matplotlib
Run: python labs/02_reward_sweep.py
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib.machinery import SourceFileLoader
m = SourceFileLoader('kicker', str(Path(__file__).with_name('01_mujoco_kicker.py'))).load_module()
import mujoco

model = mujoco.MjModel.from_xml_string(m.MODEL)
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
forces = np.linspace(0.0, 3.0, 25)
results = []
for f in forces:
    d = mujoco.MjData(model)
    for t in range(1500):
        d.ctrl[0] = f if t < 125 else -0.8 if t < 260 else 0
        mujoco.mj_step(model, d)
    xy = d.xpos[ball_id][:2].copy()
    error = float(np.linalg.norm(xy - [0.8, 0]))
    reward = np.exp(-5 * error ** 2) - 0.01 * f ** 2
    results.append((f, error, reward))
a = np.asarray(results)
print('Best strength by shaped reward:', a[np.argmax(a[:, 2])])
plt.plot(a[:, 0], a[:, 1], label='Target XY error (m)')
plt.plot(a[:, 0], a[:, 2], label='Shaped reward')
plt.xlabel('Actuator command'); plt.legend(); plt.tight_layout()
plt.savefig('reward_sweep.png', dpi=160)
print('Saved reward_sweep.png')
