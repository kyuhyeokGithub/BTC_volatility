import pandas as pd
import numpy as np
import yaml
from transformer_functions import *

from transformer_functions import final_split, final_dataload

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

with open('./model_training/configs/train.yaml') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)

conf_param = conf['params']

df=pd.read_csv("./workspace/bitcoin_price.csv")
df_norm=pd.read_csv("./workspace/bitcoin_norm.csv")


df['log_returns'] = np.log(df['Close'] /df['Close'].shift(-1))
df_norm['vol_current'] = df['log_returns'].rolling(window=2).std() * np.sqrt(365)
df_norm['vol_current'] = df_norm['vol_current'] ** (0.3)
df_norm['vol_future']=df_norm['vol_current'].shift(1)

df = df_norm
df.fillna(1, inplace=True)
df = df.rename(columns={'vol_future': 'value'})
df = df.rename(columns={'Date': 'date'})
df = df.rename(columns={'Open': 'open'})
df = df.rename(columns={'Low': 'low'})
df = df.rename(columns={'Close': 'close'})
df = df.rename(columns={'Volume': 'volume'})
df = df.rename(columns={'Marketcap': 'marketcap'})
df = df.rename(columns={'Vol/Cap': 'vol/cap'})

df['date'] = pd.to_datetime(df['date'])
df = df.set_index(['date'])
df.index = pd.to_datetime(df.index)

#btc_df = df.drop(df.index[0:2])
btc_df = df
btc_df.columns = btc_df.columns = ['btc_' + col if col not in ['value'] else col for col in btc_df.columns]
btc_df = btc_df.rename(columns={'btc_date': 'date'})


df=pd.read_csv("./workspace/ether_price.csv")
df['log_returns'] = np.log(df['Close'] /df['Close'].shift(-1))

eth_df  = pd.read_csv("./workspace/ether_norm.csv")
eth_df['vol_current'] = df['log_returns'].rolling(window=2).std() * np.sqrt(365)
eth_df['vol_current'] = eth_df['vol_current'] ** (0.3)
eth_df.columns = ['eth_' + col.lower() if col not in ['HL_spread', 'OC_spread'] else 'eth_' + col for col in eth_df.columns]
eth_df = eth_df.rename(columns={'eth_date': 'date'})
eth_df['date'] = pd.to_datetime(eth_df['date'])
eth_df = eth_df.set_index(['date'])
eth_df.index = pd.to_datetime(eth_df.index)

df=pd.read_csv("./workspace/usd_price.csv")
df['log_returns'] = np.log(df['Close'] /df['Close'].shift(-1))

usd_df  = pd.read_csv("./workspace/usdc_norm.csv")
usd_df['vol_current'] = df['log_returns'].rolling(window=2).std() * np.sqrt(365)
usd_df['vol_current'] = usd_df['vol_current'] ** (0.3)

usd_df.columns = ['usd_' + col.lower() if col not in ['HL_spread', 'OC_spread'] else 'usd_' + col for col in usd_df.columns]
usd_df = usd_df.rename(columns={'usd_date': 'date'})
usd_df['date'] = pd.to_datetime(usd_df['date'])
usd_df = usd_df.set_index(['date'])
usd_df.index = pd.to_datetime(usd_df.index)

news_df = pd.read_csv("./workspace/news_norm.csv")
news_df.columns = ['date', 'news_cnt', 'news_score_mean', 'news_score_max', 'news_score_min', 'news_score_pos_std', 'news_score_neg_std']
news_df['date'] = pd.to_datetime(news_df['date'])
news_df = news_df.set_index(['date'])
news_df.index = pd.to_datetime(news_df.index)

# btc_df  : 484 x 11(with target)
# eth_df  : 484 x 10
# usd_df  : 484 x 10
# news_df : 484 x 6

df = pd.merge(btc_df, eth_df, on='date')
df = pd.merge(df, usd_df, on='date')
df = pd.merge(df, news_df, on='date')

# df : 484 x 37 (btc,eth,usd(10x3) + news(6) + value(1,Volatility))

window = conf_param['day_window']

column_list = []
for idx in range(df.shape[1]):
    if df.columns[idx] != 'value':
        column_list.append(df.columns[idx])

if window > 1 :
    for i in range(1, window):
        for col in column_list:
            new_name = f'{i}_{col}'
            val = (-1) * i
            df[new_name] = df[col].shift(val)


#df = df.drop(columns=['news_score_pos_std', 'news_score_neg_std'])

df = df.rename(columns={'value': 'value_1'})
target_col_list = ['value_1']
for i in range(2,8) :
    new_name = f'value_{i}'
    target_col_list.append(new_name)
    df[new_name] = df['btc_vol_current'].shift(i)

df = df.drop(df.index[0:2])
df = df.drop(df.index[-31:])


def make_volatility_png():
    plot_dataframe(df)
    plot_histogram_volatility(df)

def create_dataloader(batch_size, flag):
    X_train, X_val, X_test, y_train, y_val, y_test = final_split(df, target_col_list, 0.1, 0.1)
    #print(y_train.shape, y_val.shape, y_test.shape)
    #print((y_train['value']>=1).sum())
    #print((y_val['value']>=1).sum())
    #print((y_test['value']>=1).sum())

    train_loader, valid_loader , test_loader= final_dataload(batch_size, X_train, X_val, X_test, y_train, y_val, y_test)


    if flag == 'train':
        return train_loader
    elif flag == 'valid':
        return valid_loader
    elif flag == 'test' :
        return test_loader
    else :
        return None 