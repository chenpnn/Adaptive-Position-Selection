import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import akshare as ak
from uyils.utils import normalize, func_list
import warnings
warnings.filterwarnings('ignore')


FILE_PATH = './data/raw/macro_data.xlsx'
START_DATE = '2016-01-01'
END_DATE = '2021-12-31'
EXPAND=365
print('=' * 100)

def get_macro_data(file_path, start_date, end_date, expand=None):
    start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
    if expand:
        start_date = start_date - pd.Timedelta(expand, 'd')
    df = pd.read_excel(file_path)
    print(f'Features: {list(df.columns[1:])}')
    df.columns = ['date'] + [f'mf{i+1}' for i in range(df.shape[1] - 1)]
    df['date'] = pd.to_datetime(df['date'])

    df_trade_date = ak.tool_trade_date_hist_sina()
    trade_date_list = pd.to_datetime(df_trade_date['trade_date']).values
    df = df[df['date'].isin(trade_date_list)]
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df = df.sort_values(by='date').reset_index(drop=True)
    num_null = df.isnull().sum().sum()
    if num_null == 0:
        print('There is no NAs.')
    elif num_null > 0:
        print(f'There are {num_null} NAs.')
        print('Filling NAs with forward method.')
        df = df.fillna(method='ffill')
    
    return df

df_macro = get_macro_data(FILE_PATH, START_DATE, END_DATE, expand=EXPAND)
print('Unnormalized raw macro data.')
print(df_macro.head())
print('=' * 100)

def extend_macro_data(df, func_list=[]):
    '''
    construct features based on basic features
    include: pctchange + func_list
    drop raw absolute-valued features
    '''
    df = df.copy()
    features = df.drop(columns=['date']).columns
    for func in func_list:
        df_tmp = df_macro[features].transform(func)
        df_tmp.columns = [col + func.__name__ for col in features]
        df = pd.concat([df, df_tmp], axis=1)

    return df.drop(columns = features)

df_extended_macro = extend_macro_data(df_macro, func_list)
df_extended_macro = df_extended_macro[(df_extended_macro['date'] >= START_DATE) &
                                      (df_extended_macro['date'] <= END_DATE)]

df_extended_macro = df_extended_macro.reset_index(drop=True)
print('Unnormalized extended macro data.')
print(df_extended_macro.head())
print('=' * 100)

df_extended_macro.iloc[:, 1:] = df_extended_macro.iloc[:, 1:].apply(normalize)
print('Normalized extended macro data.')
print(df_extended_macro.head())
print('=' * 100)



fig = plt.figure(figsize = (15, 5))
sns.boxplot(df_extended_macro)
plt.xticks(rotation=60)
plt.show()

df_extended_macro.to_csv('./data/processed/macro_data.csv', index=False)

