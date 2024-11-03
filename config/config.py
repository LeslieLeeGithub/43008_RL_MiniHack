import os
import torch

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 确保目录存在
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 环境配置
MAZE_SIZE = 10
MAX_EPISODE_STEPS = 100

# 训练配置分成两部分
DQN_CONFIG = {
    'learning_rate': 0.0001,
    'buffer_size': 50000,
    'learning_starts': 1000,
    'batch_size': 64,
    'gamma': 0.99,
    'exploration_fraction': 0.4,
    'exploration_final_eps': 0.05,
    'target_update_interval': 1000
}

TRAIN_CONFIG = {
    'total_timesteps': 100000,
    'log_freq': 1000
}

# 神经网络配置
POLICY_KWARGS = {
    "net_arch": [128, 128, 64]
}

# 保存配置
SAVE_FREQ = 10000

# 奖励配置
REWARD_CONFIG = {
    "win": 10.0,
    "lose": -1.0,
    "step": -0.01
}