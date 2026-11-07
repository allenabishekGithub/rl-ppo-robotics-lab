"""Export the learned SB3 PPO *deterministic policy* from Lab 03 to ONNX.
Run: python labs/04_export_onnx.py
Requires: stable-baselines3 torch onnx onnxruntime numpy
Note: illustrative Pendulum policy; NOT a deployable MicroDuck policy.
"""
import numpy as np
import torch
import onnxruntime as ort
from stable_baselines3 import PPO

agent = PPO.load('ppo_pendulum', device='cpu')
class Actor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__(); self.policy = policy
    def forward(self, obs):
        return self.policy._predict(obs, deterministic=True)

actor = Actor(agent.policy).eval()
dummy = torch.zeros((1, 3), dtype=torch.float32)
torch.onnx.export(actor, (dummy,), 'pendulum_actor.onnx',
                  input_names=['observations'], output_names=['actions'],
                  opset_version=17, dynamo=False)
obs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
with torch.no_grad(): pt = actor(torch.tensor(obs)).numpy()
session = ort.InferenceSession('pendulum_actor.onnx', providers=['CPUExecutionProvider'])
onnx_out = session.run(None, {'observations': obs})[0]
assert np.allclose(pt, onnx_out, atol=1e-4), (pt, onnx_out)
print('Export verified; PyTorch:', pt, 'ONNX:', onnx_out)
