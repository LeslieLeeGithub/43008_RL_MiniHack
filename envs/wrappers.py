import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.monitor import Monitor
import time

class MiniHackWrapper(gym.ObservationWrapper):
    """将字典观察空间转换为扁平化的Box空间"""
    def __init__(self, env):
        super().__init__(env)
        
        # 获取维度信息
        self.glyph_shape = env.observation_space["glyphs"].shape
        self.blstats_size = env.observation_space["blstats"].shape[0]
        
        # 创建扁平化的观察空间
        total_size = (np.prod(self.glyph_shape) + self.blstats_size,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=total_size,
            dtype=np.float32
        )

    def observation(self, obs):
        # 展平并合并观察
        glyphs = obs["glyphs"].astype(np.float32)
        glyphs_flat = glyphs.flatten()
        blstats = obs["blstats"].astype(np.float32)
        return np.concatenate([glyphs_flat, blstats])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

class CustomMonitor(Monitor):
    """扩展Monitor以更好地处理环境状态"""
    def __init__(self, env, filename):
        super().__init__(env, filename)
        self.needs_reset = True
        self.rewards = []
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_start_times = []
        self.total_steps = 0

    def reset(self, **kwargs):
        self.rewards = []
        self.needs_reset = False
        self.episode_start_times.append(time.time())
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        
        obs, reward, done, truncated, info = self.env.step(action)
        self.rewards.append(reward)
        
        if done or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            
        self.total_steps += 1
        return obs, reward, done, truncated, info

def make_env(log_dir):
    """创建并包装环境的工厂函数"""
    from envs.maze_env import MazeEnv
    
    env = MazeEnv()
    env = MiniHackWrapper(env)
    env = CustomMonitor(env, log_dir)
    return env