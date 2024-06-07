import pandas as pd
import numpy as np

from kw_transformer_functions import final_split, final_dataload

df=pd.read_csv("/workspace/bitcoin_price.csv")
df_norm=pd.read_csv("/workspace/bitcoin_norm.csv")
df['log_returns'] = np.log(df['close'] /df['close'].shift(1))
df_norm['vol_current'] = df['log_returns'].rolling(window=2).std() * np.sqrt(365)
df_norm['vol_future']=df_norm['vol_current'].shift(-1)
df = df_norm
df.fillna(1, inplace=True)
df = df.rename(columns={'vol_future': 'value'})
df = df.rename(columns={'Date': 'date'})
df['date'] = pd.to_datetime(df['date'])
df = df.set_index(['date'])
df.index = pd.to_datetime(df.index)

def create_dataloader(batch_size, cv):
    X_train, X_test, y_train, y_test = final_split(df, 'value', 0.1)

    train_loader, test_loader, test_loader_one, scaler = final_dataload(batch_size, X_train, X_test, y_train, y_test)

    if cv == 0:
        return train_loader
    else:
        return test_loader 