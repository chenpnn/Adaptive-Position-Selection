import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradeEnv(gym.Env):
    """A custom stock trading environment based on OpenAI gymnasium"""
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 df_macro,
                 df_factor,
                 df_stock,
        ):
        super(TradeEnv, self).__init__()

        self.df_macro = df_macro
        self.df_factor = df_factor
        self.df_stock = df_stock


    def _next_observation(self):
        pass

    def _take_action(self, action):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass