import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradeEnv(gym.Env):
    """A custom stock trading environment based on OpenAI gymnasium"""
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 df_macro,
                 df_stock,
                 df_factor,
                 look_forward=1,
                 fee_buy = 0.001,
                 fee_sell = 0.001
        ):
        '''
        - df_macro:
        - df_stock:
        - df_factor:
        - look_forward: define reward as the return of the future look_forward days
        - fee: bid fee
        '''
        super(TradeEnv, self).__init__()
        
        print('Initialize trade environment...')
        print('Aligning df_macro, df_stock and df_factor...')
        self.dates = list(set(df_macro['date']) & set(df_factor['date']) & set(df_stock['date']))
        self.cols = list(set(df_factor.columns) & set(df_stock.columns))
        self.data_macro = df_macro.loc[df_macro['date'].isin(self.dates)].set_index('date').values
        self.data_factor = df_factor.loc[df_factor['date'].isin(self.dates), self.cols].set_index('date').values
        self.data_stock = df_stock.loc[df_stock['date'].isin(self.dates), self.cols].set_index('date').values
        print('df_macro shape: ', self.data_macro.shape)
        print('df_stock shape: ', self.data_stock.shape)
        print('df_factor shape: ', self.data_factor.shape)

        self.look_forward = look_forward
        self.fee_buy = fee_buy
        self.fee_sell = fee_sell

        self.current_idx = 0
        self.max_idx = len(self.data_macro)-1
        self.terminated  = False


    def _to_weight(self, factor, action):
        '''
        trasform a factor array to a weight array via
        $w = tanh(scale * max(factor - threshold, 0))$
        $w = w / sum(w)$

        - factor: shape=(N,)
        - action: action = [scale, threshold], shape=(2,)
        '''
        # the larger the scale, more concentrated the weights
        scale, threshold = action
        factor = np.array(factor)
        w = np.tanh(scale * np.maximum(factor-threshold, 0))
        w /= np.sum(w)
        return w
    
    # important
    def step(self, action):
        if self.current_idx + self.look_forward > self.max_idx:
            self.current_idx = 0
            self.terminated  = True
        else:
            factor = self.data_factor[self.current_idx]
            factor = (factor - np.mean(factor)) / np.std(factor)  # scale the factor array
            w = self._to_weight(factor, action)
            ret = self.data_stock[self.current_idx + self.look_forward] / self.data_stock[self.current_idx] - 1
            self.observation = [ret, factor]
            self.reward = self.observation[0] @ w - self.fee

            self.current_idx += 1
        return self.observation, self.reward, self.terminated


    # important
    def reset(self):
        pass
    
    # optional
    def render(self, mode='human', close=False):
        pass

    # optional
    def close(self):
        pass


if __name__ == '__main__':
    file_path = '../data/processed/'
    df_macro = pd.read_csv(file_path + 'macro_data.csv')
    df_stock = pd.read_csv(file_path + 'stock_data.csv')
    df_factor = pd.read_csv(file_path + 'factor_data.csv')

    env = TradeEnv(df_macro=df_macro, df_factor=df_factor, df_stock=df_stock)

    env.data_stock
    env.current_idx
    obs, reward, term = env.step([1, 2])

    for _ in range(10):
        obs, reward, term = env.step([1, 2])
        print(env.current_idx, reward)

    