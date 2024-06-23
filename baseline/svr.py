# -*- coding: utf-8 -*-
"""SVR.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wtQr0qRYsDCdzIiivubRsqdsC278Vumj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    # SVR 모델 하이퍼파라미터 그리드 설정
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'epsilon': [0.01, 0.1, 1]
    }

    # GridSearchCV를 사용하여 최적의 파라미터 탐색
    grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # 최적의 모델로 예측
    best_svr_model = grid_search.best_estimator_

    # 검증 데이터 예측
    validation_predict = best_svr_model.predict(X_validation_scaled)

    # 예측값 시각화 (Validation)
    plt.figure(figsize=(10, 6))
    plt.plot(validation.index, y_validation, label='Actual Volatility')
    plt.plot(validation.index, validation_predict, label='Predicted Volatility')
    plt.title('Bitcoin Volatility Forecast vs Actuals (Validation)')
    plt.legend()
    plt.show()

    # 검증 데이터 예측 성능 평가
    validation_rmse = np.sqrt(mean_squared_error(y_validation, validation_predict))
    print(f'Validation RMSE: {validation_rmse}')

    # 전체 테스트 데이터 예측
    test_predict_all = best_svr_model.predict(X_test_scaled)

    # 성능 평가 지표 계산 함수
    def calculate_metrics(actual, predicted):
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        return rmse, mae, mape

    # 성능 평가
    test_rmse, test_mae, test_mape = calculate_metrics(y_test, test_predict_all)

    print(f'Test RMSE: {test_rmse}')
    print(f'Test MAE: {test_mae}')
    print(f'Test MAPE: {test_mape}%')

    # 예측값 시각화 (Test, 전체 기간)
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, y_test, label='Actual Volatility')
    plt.plot(test.index, test_predict_all, label='Predicted Volatility')
    plt.title('Bitcoin Volatility Forecast vs Actuals (Test, 전체 기간)')
    plt.legend()
    plt.show()

    # 새로운 손실값을 계산하고 성능 평가 지표 재계산
    def calculate_new_metrics(actual, predicted):
        actual_new = actual ** (10/3)
        predicted_new = predicted ** (10/3)
        new_rmse = np.sqrt(mean_squared_error(actual_new, predicted_new))
        new_mae = mean_absolute_error(actual_new, predicted_new)
        new_mape = np.mean(np.abs((actual_new - predicted_new) / actual_new)) * 100
        return new_rmse, new_mae, new_mape

    # 검증 데이터에서 새로운 성능 평가
    validation_new_rmse, validation_new_mae, validation_new_mape = calculate_new_metrics(y_validation, validation_predict)

    print(f'New Validation RMSE: {validation_new_rmse}')
    print(f'New Validation MAE: {validation_new_mae}')
    print(f'New Validation MAPE: {validation_new_mape}%')

    # 테스트 데이터에서 새로운 성능 평가
    test_new_rmse, test_new_mae, test_new_mape = calculate_new_metrics(y_test, test_predict_all)

    print(f'New Test RMSE: {test_new_rmse}')
    print(f'New Test MAE: {test_new_mae}')
    print(f'New Test MAPE: {test_new_mape}%')

    # 새로운 예측값 시각화 (Validation)
    plt.figure(figsize=(10, 6))
    plt.plot(validation.index, y_validation ** (10/3), label='Actual Volatility (10/3)')
    plt.plot(validation.index, validation_predict ** (10/3), label='Predicted Volatility (10/3)')
    plt.title('Bitcoin Volatility Forecast vs Actuals (Validation, 10/3)')
    plt.legend()
    plt.show()

    # 새로운 예측값 시각화 (Test, 전체 기간)
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, y_test ** (10/3), label='Actual Volatility (10/3)')
    plt.plot(test.index, test_predict_all ** (10/3), label='Predicted Volatility (10/3)')
    plt.title('Bitcoin Volatility Forecast vs Actuals (Test)')
    plt.legend()
    plt.show()

    print(best_svr_model)

if __name__ == "__main__":
    run_model()