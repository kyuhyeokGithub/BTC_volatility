# test.py

import grach
import svr
import randomforest
import gradientboostingmachine
import lstm

def main():
    print("Running Grach Model...")
    grach.run_model()  # grach.py 파일의 모델 실행 함수 호출
    
    print("Running SVR Model...")
    svr.run_model()  # svr.py 파일의 모델 실행 함수 호출
    
    print("Running Random Forest Model...")
    randomforest.run_model()  # randomforest.py 파일의 모델 실행 함수 호출
    
    print("Running Gradient Boosting Machine Model...")
    gradientboostingmachine.run_model()  # gradientboostingmachine.py 파일의 모델 실행 함수 호출
    
    print("Running LSTM Model...")
    lstm.run_model()  # lstm.py 파일의 모델 실행 함수 호출

if __name__ == "__main__":
    main()
