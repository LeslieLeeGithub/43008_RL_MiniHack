import sys
import os

# 将项目根目录添加到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # tests 目录
project_root = os.path.dirname(current_dir)               # 项目根目录
sys.path.append(project_root)

from config.config import LOG_DIR
from envs.wrappers import make_env

def test_environment(env, num_episodes=1):
    """测试环境的基本功能"""
    print("\nTesting environment...")
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}")
        try:
            obs, info = env.reset()
            print(f"Reset successful. Observation shape: {obs.shape}")
            
            episode_reward = 0
            step_count = 0
            
            while True:
                action = env.action_space.sample()
                print(f"\nAttempting action {action}")
                
                try:
                    obs, reward, done, truncated, info = env.step(action)
                    episode_reward += reward
                    step_count += 1
                    print(f"Step {step_count} completed")
                    print(f"- Reward: {reward:.2f}")
                    print(f"- Done: {done}")
                    print(f"- Total reward: {episode_reward:.2f}")
                    
                    if done or truncated:
                        print(f"\nEpisode finished after {step_count} steps.")
                        print(f"Total reward: {episode_reward:.2f}")
                        break
                        
                except Exception as e:
                    print(f"Error during step: {e}")
                    return False
                
                if step_count >= 100:
                    print("\nReached maximum steps")
                    break
                    
        except Exception as e:
            print(f"Error during episode: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print(f"Project root path: {project_root}")  # 调试用
    print(f"Python path: {sys.path}")           # 调试用
    
    # 创建环境
    env = make_env(LOG_DIR)
    print("\nEnvironment created.")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # 运行测试
    success = test_environment(env, num_episodes=2)
    print("\nTest result:", "Success" if success else "Failed")
    
    # 清理
    env.close()