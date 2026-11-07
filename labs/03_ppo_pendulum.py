"""Learn PPO basics on Pendulum-v1 (CPU friendly).
Run: python labs/03_ppo_pendulum.py
Requires: gymnasium stable-baselines3 torch
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = Monitor(gym.make('Pendulum-v1'))
model = PPO('MlpPolicy', env, verbose=1, seed=42, n_steps=1024,
            batch_size=64, learning_rate=3e-4, gamma=0.99,
            gae_lambda=0.95, clip_range=0.2)
model.learn(total_timesteps=30000)
model.save('ppo_pendulum')
obs, info = env.reset(seed=123)
score = 0.0
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    score += reward
    if terminated or truncated: break
print(f'Evaluation episode reward: {score:.2f}')
env.close()
