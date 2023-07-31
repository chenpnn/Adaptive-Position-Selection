import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

#AlphaEnv:      Factors -[AlphaMod]-> Alpha -[AlphaEnv]-> Weight (Stock)  
#
#   Factor: 股票因子，各种收益预测模型的输出，Factor的构建与计算是env之前的环节
#   Alpha:  Factor的函数，将各种Factor合成一个股票打分。Factor合成Alpha是env之前的环节
#   Weight：投资组合股票权重

class AlphaEnv(gym.Env):
    """A custom stock trading environment based on OpenAI gymnasium"""
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 df_macro,
                 df_stock,
                 df_alpha,
                 num_actions=2,
                 look_forward=1,
                 fee_ratio=2e-4,
                 trade_pct=0.5   #交易比例，取值范围(0,1]
        ):
        '''
        - df_macro:
        - df_stock:
        - df_alpha:
        - look_forward: define reward as the return of the future look_forward days
        - fee_ratio: bid fee ratio
        '''
        super(AlphaEnv, self).__init__()
        
        assert 1 == 1  # 验证三个df日期和代码是否对齐

        # print('Initialize trade environment...')
        # print('Aligning df_macro, df_stock and df_alpha...')

        # self.dates = list(set(df_macro['date']) & set(df_alpha['date']) & set(df_stock['date']))
        # self.cols = list(set(df_alpha.columns) & set(df_stock.columns))
        # self.df_macro = df_macro.loc[df_macro['date'].isin(self.dates)].set_index('date').values
        # self.data_alpha = df_alpha.loc[df_alpha['date'].isin(self.dates), self.cols].set_index('date').values
        # self.df_stock = df_stock.loc[df_stock['date'].isin(self.dates), self.cols].set_index('date').values

        self.df_macro = df_macro
        self.df_stock = df_stock
        self.df_alpha = df_alpha

        self.num_periods, self.num_stocks = self.df_stock.shape
        self.num_macros = self.df_macro.shape[1] - 1
        self.num_actions = num_actions

        # print('df_macro shape: ', self.df_macro.shape)
        # print('df_stock shape: ', self.df_stock.shape)
        # print('df_alpha shape: ', self.data_alpha.shape)

        self.col_date, self.col_code, self.col_alpha = df_alpha.columns  # 提取列名
        self.col_return = df_stock.columns[-1]  # df_stock最后一列为收益率
        self.col_macro = df_macro.columns[1:]  # df_macro第一列到最后一列都是因子值

        self.look_forward = look_forward
        self.fee_ratio = fee_ratio
        self.trade_pct = trade_pct

        # self.current_idx = 0
        self.max_idx = len(self.df_macro)-1
        # self.terminated  = False

    # weight_alpha：模型权重w_a，是关于alpha的单调函数，且L1范数为1
    def weight_alpha(self, alpha, action):
        alpha = np.array(alpha)
        scale, threshold = action[0], action[1]
        # w = np.tanh(np.maximum(np.array(alpha) - threshold, 0) + threshold) ** scale
        # w[w < np.tanh(threshold * scale)] = 0
        # w /= np.sum(w)
        w = np.sign(alpha) * np.abs(alpha) ** scale
        w[w < threshold] = 0
        # w[np.nonzero(w)[0]] = w[np.nonzero(w)[0]] ** scale
        # w = np.sign(w) * np.abs(w) ** scale
        # w[w <= np.tanh(threshold) * scale] = 0
        if np.sum(w) == 0:
            w = np.ones_like(w) / len(w)
        else:
            w /= np.sum(w)
        return w
    
    # weight_target: 目标权重w_t，是w_0与w_a的加权平均,同时产生交易权重w_delta
    def weight_target(self, w_0, w_a):
        l = self.trade_pct
        w = l * w_a + (1-l) * w_0
        w_delta = l * np.abs(w_a - w_0)
        return w, w_delta
    
    # weight_end：结算权重w_end，w_t经过日内收益变化，再减去交易费用，最后归一化
    def weight_end(self, r, w_t, w_delta):
        fee = np.sum(self.fee_ratio * w_delta)
        #收盘时，权重由于股票的日内收益已经产生变化，同时收盘前要按市值等比例卖出还交易费用
        w_end = (1 + r) * w_t
        w_sum_end = np.sum(w_end)
        r_end = w_sum_end - fee - 1
        #等比例减仓
        w = w_end * (1 + r_end) / w_sum_end
        #归一化
        w /= np.sum(w)
        return w, r_end, fee

    
    # important
    def step(self, action):
        if self.current_idx + self.look_forward + 1 > self.max_idx:
            self.current_idx = 0
            self.terminated  = True
            return None, None, True
        else:
            #reward计算
            #1.获得t日数据，每股的alpha和收益率（r）
            #2.获得模型权重w_a，根据action和alpha计算
            #3.获得目标权重w_t，根据超参数-调参比例l和w_a及t日起始权重w_start计算
            #4.获得结算权重w_end，同时获得reward和交易费用fee
            #5.更新t+1日起始权重w_start（就是t日的w_end）
            alpha = self.df_alpha[self.current_idx]
            r = self.df_stock[self.current_idx + self.look_forward + 1] / self.df_stock[self.current_idx + 1] - 1
            w_a = self.weight_alpha(alpha, action)
            w_t, w_delta = self.weight_target(self.w_start, w_a)
            w_end, reward, fee = self.weight_end(r, w_t, w_delta)
            self.w_start = w_end
            
            #next state
            self.current_idx += 1 
            next_state = self.df_macro.iloc[self.current_idx, 1:]
            return next_state, reward, False
    

    def predict(self,
                df_stock, 
                df_alpha, 
                action=[0, 1]):
        '''
        预测收益率，给定动作值
        '''
        returns = {}
        action = np.array(action)
        col_date, col_code, col_alpha = df_alpha.columns  # 提取列名
        col_return = df_stock.columns[-1]
        dates = df_alpha.sort_values(col_date)[col_date].unique()  # 第一列为交易日期
        for i in range(len(dates) - 1):
            date_now = dates[i]
            date_next = dates[i+1]
            alpha = df_alpha[df_alpha[col_date] == date_now][col_alpha]  # 当前交易日的因子值
            weight = self.weight_alpha(alpha, action)  # 预测的权重
            returns[date_next] = df_stock[df_stock[col_date] == date_next][col_return].values @ weight  # 预测收益率
        return returns
        


    # important
    def reset(self):
        self.current_idx = 0
        self.w_start = np.zeros(self.df_stock.shape[1])
        return self.df_macro.iloc[0, 1:].values


if __name__ == '__main__':
    # file_path = '../data/processed/'
    # # file_path ='C:/alib/research/AI_E2E/Optimization/main/model/DDPG/data/processed/'
    # df_macro = pd.read_csv(file_path + 'macro_data.csv')
    # df_stock = pd.read_csv(file_path + 'stock_data.csv')
    # df_alpha = pd.read_csv(file_path + 'factor_data.csv')

    # env = AlphaEnv(df_macro,df_stock,df_alpha)
    # env.reset()
    # env.df_stock
    # env.current_idx
    # obs, reward, term = env.step([1, 2])

    # for _ in range(10):
    #     obs, reward, term = env.step([1, 2])
    #     print(env.current_idx, reward)

    


