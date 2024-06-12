import torch
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
from plotly.offline import iplot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def feature_label_split(df, target_col_list):
    y = df[target_col_list]
    X = df.drop(columns=target_col_list)
    return X, y

def train_val_test_split(df, target_col_list, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col_list)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

def final_split(df, target_col_list, val_ratio, test_ratio):
    X, y = feature_label_split(df, target_col_list)
    X_test, X_tmp, y_test, y_tmp = train_test_split(X, y, test_size=1-test_ratio, shuffle=False)
    X_val, X_train, y_val, y_train = train_test_split(X_tmp, y_tmp, test_size=((1-val_ratio-test_ratio)/(1-test_ratio)), shuffle=False)

    return X_train, X_val, X_test, y_train, y_val, y_test

def final_dataload(batch_size, X_train, X_val, X_test, y_train, y_val, y_test):
   
    train_features = torch.Tensor(X_train.values)
    train_targets = torch.Tensor(y_train.values)

    val_features = torch.Tensor(X_val.values)
    val_targets = torch.Tensor(y_val.values)

    test_features = torch.Tensor(X_test.values)
    test_targets = torch.Tensor(y_test.values)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)
    valid_loader = DataLoader(val, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=1, num_workers=8, shuffle=False, drop_last=True)
    
    return train_loader, valid_loader , test_loader


def plot_predictions(df_result):
    data = []
    
    value = go.Scatter(
        x=df_result.index,
        y=df_result.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df_result.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    
    prediction = go.Scatter(
        x=df_result.index,
        y=df_result.prediction,
        mode="lines",
        line={"dash": "dot"},
        name='predictions',
        marker=dict(),
        text=df_result.index,
        opacity=0.8,
    )
    data.append(prediction)
    
    layout = dict(
        title="Predictions vs Actual Values for the dataset",
        xaxis=dict(title="Time", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)

def plot_dataset(df, title):
    data = []
    value = go.Scatter(
        x=df.index,
        y=df.value,
        mode="markers",
        name="texts",
        marker=dict(),
        text=df.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    layout = dict(
        title=title,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)
    
    
def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
        #print(df[col])
    return df


def format_predictions(predictions, values, df_test, scaler):

    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result        

def calculate_metrics(df):
    return {'mae' : mean_absolute_error(df.value, df.prediction),
            'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
            'r2' : r2_score(df.value, df.prediction),
           'rmspe' : np.sqrt(np.mean(np.square(((df.value - df.prediction) / df.value)), axis=0))
           }

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def RMSPELoss(y_pred, y_true):
    return torch.sqrt(torch.mean( ((y_true - y_pred) / y_true) ** 2 ))

def MAELoss(yhat, y):
    return torch.mean(torch.abs(yhat - y))

def MAPELoss(yhat, y):
    return torch.mean(torch.abs((yhat - y) / y))


def plot_dataframe(df) :
    plt.figure(figsize = (10, 5))
    plt.plot(df.index, df['value_1']**(10/3), linestyle='-', color='blue', marker='o', label='Volatility before Transformation')
    plt.plot(df.index, df['value_1'], linestyle='-', color='red', marker = 'x', label='Volatility after Transformation')
    plt.title('Volatility change after Transformation')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc='upper right')
    plt.grid(True)
    #plt.show()
    plt.savefig('./plot_volatility.png')

def plot_histogram_volatility(df) :

    bin_width = 0.025
    bins = np.arange(0, 2+bin_width, bin_width)

    plt.figure(figsize = (10, 5))

    plt.hist(df['value_1']**(10/3), bins=bins, alpha = 0.5, label='Volatility before Transformation', color = 'blue', edgecolor = 'k')
    plt.hist(df['value_1'], bins=bins, alpha = 0.5, label='Volatility after Transformation', color = 'red', edgecolor = 'k')

    plt.title('Overlayed Histograms for Volatility Transformation')
    plt.xlabel('Volatility')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True)
    #plt.show()
    plt.savefig('./overlayed_histogram_volatility.png')