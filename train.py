import os
import time
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
from config.config import (
    DEVICE, LOG_DIR, MODELS_DIR, DQN_CONFIG, 
    TRAIN_CONFIG, POLICY_KWARGS
)
from envs.wrappers import make_env

class TrainingMonitor(BaseCallback):
    """
    跟踪训练进度的回调函数，提供详细的训练信息
    """
    def __init__(self, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.last_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_episodes = 0
        
    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            # 获取最新完成的回合信息
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info['r'])
            self.episode_lengths.append(ep_info['l'])
            self.total_episodes = len(self.episode_rewards)
            
            # 每 check_freq 步打印一次详细信息
            if self.n_calls % self.check_freq == 0:
                # 计算时间和速度指标
                current_time = time.time()
                time_elapsed = current_time - self.last_time
                steps_per_second = self.check_freq / time_elapsed
                
                # 计算最近的统计信息
                recent_rewards = self.episode_rewards[-10:]  # 最近10个回合
                mean_reward = np.mean(recent_rewards) if recent_rewards else 0
                std_reward = np.std(recent_rewards) if recent_rewards else 0
                
                # 打印详细信息
                print("\n====== Training Update ======")
                print(f"Total timesteps: {self.n_calls}")
                print(f"Episodes completed: {self.total_episodes}")
                print(f"Mean reward (last 10 ep): {mean_reward:.2f} ± {std_reward:.2f}")
                print(f"Best mean reward: {self.best_mean_reward:.2f}")
                print(f"Steps per second: {steps_per_second:.2f}")
                
                # 更新最佳得分并保存模型
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
                
                # 打印探索信息
                exploration_rate = self.model.exploration_rate
                print(f"Current exploration rate: {exploration_rate:.3f}")
                
                # 打印最近一个回合的详细信息
                print("\nLatest episode stats:")
                print(f"Length: {self.episode_lengths[-1]}")
                print(f"Reward: {self.episode_rewards[-1]:.2f}")
                print("===========================\n")
                
                self.last_time = current_time
                
                # 绘制学习曲线
                self._plot_learning_curve()
        
        return True
    
    def _plot_learning_curve(self):
        """绘制并保存学习曲线"""
        plt.figure(figsize=(10, 5))
        
        # 平滑化reward曲线
        rewards = np.array(self.episode_rewards)
        window = min(10, len(rewards))
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        plt.plot(smoothed_rewards)
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Reward')
        plt.savefig(os.path.join(self.log_dir, 'learning_curve.png'))
        plt.close()

def train_agent(total_timesteps=None, log_freq=1000):
    """
    训练智能体
    
    Args:
        total_timesteps (int): 总训练步数
        log_freq (int): 日志记录频率
    """
    print("====== Starting Training ======")
    print(f"Device: {DEVICE}")
    print(f"Total timesteps: {total_timesteps or TRAIN_CONFIG['total_timesteps']}")
    print("DQN configuration:", DQN_CONFIG)
    print("Policy architecture:", POLICY_KWARGS)
    print("==============================\n")
    
    # 创建环境
    env = make_env(LOG_DIR)
    print("\nEnvironment created successfully.")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # 创建模型
    model = DQN(
        env=env,
        policy="MlpPolicy",
        verbose=1,
        device=DEVICE,
        **DQN_CONFIG,  # 只传入DQN的配置参数
        policy_kwargs=POLICY_KWARGS
    )
    
    # 创建训练监视器
    monitor = TrainingMonitor(
        check_freq=log_freq,
        log_dir=LOG_DIR,
        verbose=1
    )
    
    try:
        print("\nStarting training process...")
        model.learn(
            total_timesteps=total_timesteps or TRAIN_CONFIG['total_timesteps'],
            callback=monitor,
            progress_bar=True
        )
        
        # 保存最终模型
        final_model_path = os.path.join(MODELS_DIR, "final_model")
        model.save(final_model_path)
        print(f"\nTraining completed! Final model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        print("Saving current model...")
        model.save(os.path.join(MODELS_DIR, "interrupted_model"))
        
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
        
    finally:
        env.close()
        print("\nEnvironment closed.")
        
    return model

def evaluate_agent(model, num_episodes=10):
    """
    评估训练好的智能体
    
    Args:
        model: 训练好的模型
        num_episodes (int): 评估回合数
    """
    env = make_env(LOG_DIR)
    
    print("\n====== Starting Evaluation ======")
    rewards = []
    lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        rewards.append(episode_reward)
        lengths.append(steps)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    print("\nEvaluation Results:")
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean episode length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
    print("=============================")
    
    env.close()
    return np.mean(rewards), np.std(rewards)

if __name__ == "__main__":
    # 训练智能体
    model = train_agent(
        total_timesteps=TRAIN_CONFIG['total_timesteps'],
        log_freq=TRAIN_CONFIG['log_freq']
    )
    
    # 评估智能体
    if model is not None:
        evaluate_agent(model, num_episodes=5)