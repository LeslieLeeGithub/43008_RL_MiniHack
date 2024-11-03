import numpy as np
from minihack import LevelGenerator
from nle.nethack import CompassDirection
import gymnasium as gym
from gymnasium import spaces
from config.config import MAZE_SIZE, MAX_EPISODE_STEPS, REWARD_CONFIG
import minihack
from nle import nethack

class MazeGenerator:
    def __init__(self, size=MAZE_SIZE):
        self.size = size
        self.lg = LevelGenerator(w=size, h=size)
        self._generate_maze()
        
    def _generate_maze(self):
        # 填充地面
        self.lg.fill_terrain("fillrect", ".", 0, 0, self.size-1, self.size-1)
        
        # 添加边界墙壁
        for x in range(self.size):
            self.lg.add_terrain((x, 0), "#")
            self.lg.add_terrain((x, self.size-1), "#")
        for y in range(self.size):
            self.lg.add_terrain((0, y), "#")
            self.lg.add_terrain((self.size-1, y), "#")
            
        # 添加内部墙壁
        walls = [
            (3, 3), (3, 4), (3, 5), (3, 6),
            (6, 3), (6, 4), (6, 5), (6, 6),
            (4, 3), (5, 3),
            (4, 6), (5, 6)
        ]
        for x, y in walls:
            self.lg.add_terrain((x, y), "#")
            
        # 设置起点和终点
        self.lg.set_start_pos((1, 1))
        self.lg.add_goal_pos((8, 8))
        
    def get_des(self):
        return self.lg.get_des()

class MazeEnv(gym.Env):
    """Custom NetHack maze environment."""
    def __init__(self):
        super().__init__()
        
        # 生成迷宫
        self.maze_generator = MazeGenerator()
        
        # 首先定义基础动作
        self.base_actions = [
            CompassDirection.N,   # 北
            CompassDirection.E,   # 东
            CompassDirection.S,   # 南
            CompassDirection.W,   # 西
            CompassDirection.NE,  # 东北
            CompassDirection.SE,  # 东南
            CompassDirection.SW,  # 西南
            CompassDirection.NW,  # 西北
        ]
        
        # 创建内部环境，显式指定可用动作
        self._env = minihack.MiniHackNavigation(
            des_file=self.maze_generator.get_des(),
            max_episode_steps=MAX_EPISODE_STEPS,
            observation_keys=("glyphs", "chars", "colors", "specials", "blstats"),
            actions=self.base_actions,  # 传入动作列表
        )
        
        # 设置动作空间
        self.action_space = spaces.Discrete(len(self.base_actions))
        
        # 设置观察空间
        spaces_dict = {
            "glyphs": spaces.Box(
                low=0,
                high=255,
                shape=self._env.observation_space["glyphs"].shape,
                dtype=np.uint8
            ),
            "blstats": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self._env.observation_space["blstats"].shape,
                dtype=np.int32
            ),
            "chars": spaces.Box(
                low=0,
                high=255,
                shape=self._env.observation_space["chars"].shape,
                dtype=np.uint8
            ),
            "colors": spaces.Box(
                low=0,
                high=255,
                shape=self._env.observation_space["colors"].shape,
                dtype=np.uint8
            ),
            "specials": spaces.Box(
                low=0,
                high=255,
                shape=self._env.observation_space["specials"].shape,
                dtype=np.uint8
            )
        }
        self.observation_space = spaces.Dict(spaces_dict)
        
        print("MazeEnv initialized with actions:", [a.name for a in self.base_actions])
    
    def step(self, action):
        """Execute one time step within the environment."""
        try:
            # 确保动作在有效范围内
            if not 0 <= action < len(self.base_actions):
                raise ValueError(f"Invalid action {action}")
            
            # 直接使用动作索引
            print(f"Executing action {action} ({self.base_actions[action].name})")
            
            # 执行动作
            obs, reward, done, info = self._env.step(action)  # 直接传递索引
            
            # 调整奖励
            if done and reward > 0:
                reward = REWARD_CONFIG["win"]
                print(f"Won! Reward: {reward}")
            elif reward < 0:
                reward = REWARD_CONFIG["lose"]
                print(f"Lost. Reward: {reward}")
            else:
                reward = REWARD_CONFIG["step"]
                print(f"Step. Reward: {reward}")
            
            return obs, reward, done, False, info
            
        except Exception as e:
            print(f"Error in step: action={action}, error={str(e)}")
            raise

    def reset(self, *, seed=None, options=None):
        """Reset the environment to an initial state."""
        try:
            obs = self._env.reset()
            print("Environment reset successful")
            return obs, {}
        except Exception as e:
            print(f"Error in reset: {str(e)}")
            raise
    
    def render(self):
        """Render the current environment state."""
        try:
            return self._env.render()
        except Exception as e:
            print(f"Error in render: {str(e)}")
            raise
    
    def close(self):
        """Clean up resources."""
        try:
            self._env.close()
            print("Environment closed successfully")
        except Exception as e:
            print(f"Error in close: {str(e)}")
            raise