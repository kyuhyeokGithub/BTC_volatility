# -*- coding: utf-8 -*-
"""lstm.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gT2KPoQ75G1Y6qB6Bs6QbRka0mi_0ZNt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def run_model():
    # 파일 경로 설정
    file_path_news_df = './workspace/news_norm.csv'
    file_path_usdc_df = './workspace/usdc_norm.csv'
    file_path_bitcoin_norm = "./workspace/bitcoin_norm.csv"
    file_path_ether_df = './workspace/ether_norm.csv'
    file_path_bitcoin_df = "./workspace/bitcoin_price.csv"

    # 데이터 불러오기 및 컬럼명 변경
    def load_and_rename(file_path, prefix):
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        df = df.add_prefix(prefix + '_')
        return df

    news_df = load_and_rename(file_path_news_df, 'news')
    usdc_df = load_and_rename(file_path_usdc_df, 'usdc')
    bitcoin_norm = load_and_rename(file_path_bitcoin_df, 'bitcoin') #df_norm
    ether_df = load_and_rename(file_path_ether_df, 'ether')
    bitcoin_df = load_and_rename(file_path_bitcoin_df, 'bitcoin_r') #not normalized #df

    # 로그 수익률 및 변동성 계산
    bitcoin_df['bitcoin_log_returns'] = np.log(bitcoin_df['bitcoin_r_Close'] / bitcoin_df['bitcoin_r_Close'].shift(-1))
    bitcoin_norm['bitcoin_volatility'] = bitcoin_df['bitcoin_log_returns'].rolling(window=2).std() * np.sqrt(365)
    bitcoin_norm['bitcoin_volatility'] = bitcoin_norm['bitcoin_volatility'] ** (0.3)

    # 타겟 변수 설정
    merged_df = pd.concat([news_df, usdc_df, bitcoin_norm, ether_df], axis=1).dropna()
    merged_df['bitcoin_volatility'] = bitcoin_norm['bitcoin_volatility']

    # DatetimeIndex 정렬
    merged_df = merged_df.sort_index()

    # 데이터 분할
    train = merged_df['2021-02-01':'2023-09-05']
    validation = merged_df['2023-09-06':'2024-01-01']
    test = merged_df['2024-01-02':'2024-04-28']

    # 독립 변수와 종속 변수 분리
    X_train = train.drop(columns=['bitcoin_volatility'])
    y_train = train['bitcoin_volatility']
    X_validation = validation.drop(columns=['bitcoin_volatility'])
    y_validation = validation['bitcoin_volatility']
    X_test = test.drop(columns=['bitcoin_volatility'])
    y_test = test['bitcoin_volatility']

    # 데이터 스케일링
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    # 데이터셋 준비 함수
    def create_dataset(dataset, target, look_back=10):
        X, Y = []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), :]
            X.append(a)
            Y.append(target[i + look_back])
        return np.array(X), np.array(Y)

    look_back = 10
    X_train_lstm, y_train_lstm = create_dataset(X_train_scaled, y_train.values, look_back)
    X_validation_lstm, y_validation_lstm = create_dataset(X_validation_scaled, y_validation.values, look_back)
    X_test_lstm, y_test_lstm = create_dataset(X_test_scaled, y_test.values, look_back)

    # LSTM 모델 구축
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(look_back, X_train_lstm.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    model.summary()

    # 조기 종료 설정
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # 모델 학습
    history = model.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=32, validation_data=(X_validation_lstm, y_validation_lstm), verbose=2, callbacks=[early_stop])

    # 학습 결과 시각화
    plt.figure(figsize=(14, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 테스트 데이터로 예측
    y_pred = model.predict(X_test_lstm)
    # 예측 데이터와 실제 데이터를 10/3 제곱으로 변환
    y_test_lstm_transformed = y_test_lstm ** (10 / 3)
    y_pred_transformed = y_pred ** (10 / 3)

    # 예측 결과 평가
    rmse = np.sqrt(mean_squared_error(y_test_lstm_transformed, y_pred_transformed))
    mae = mean_absolute_error(y_test_lstm_transformed, y_pred_transformed)
    mape = mean_absolute_percentage_error(y_test_lstm_transformed, y_pred_transformed)

    print(f'Test RMSE: {rmse}')
    print(f'Test MAE: {mae}')
    print(f'Test MAPE: {mape}')

    # 예측 결과 시각화
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_lstm_transformed, label='True Volatility')
    plt.plot(y_pred_transformed, label='Predicted Volatility')
    plt.title('True vs Predicted Volatility (Transformed)')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_model()
