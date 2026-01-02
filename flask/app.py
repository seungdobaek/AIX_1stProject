from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import warnings
import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings('ignore')

def call_api():
    url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
    key = os.getenv('serviceKey')

    now = datetime.now()
    if ( now.hour < 3):
        now = now-timedelta(days=1)
    base_date = now.strftime('%Y%m%d')
    hour = now.hour
    if ( now.minute <= 15):
        hour = hour-1
    base_date = now.strftime('%Y%m%d')
    hour = (hour-2)//3*3+2
    if (hour < 0):
        hour = hour+24
    base_time = '{}00'.format(hour)

    params ={'serviceKey' : key, 'pageNo' : '1', 'numOfRows' : '1200', 'dataType' : 'XML', 'base_date' : base_date, 'base_time' : base_time, 'nx' : '60', 'ny' : '127' }

    response = requests.get(url, params=params)
    soup = BeautifulSoup(response.text, 'xml')
    return soup.select('item')

item_list = call_api()
print(item_list)

app = Flask(__name__)

# 모델 로드
try:
    model_forecast = joblib.load('models/예보모델.joblib')
    print("✓ 예보모델.joblib 로드 완료")
except Exception as e:
    print(f"✗ 예보모델.joblib 로드 실패: {e}")
    model_forecast = None

try:
    model_lgbm = joblib.load('models/LGBM_model.pkl')
    print("✓ LGBM_model.pkl 로드 완료")
except Exception as e:
    print(f"✗ LGBM_model.pkl 로드 실패: {e}")
    model_lgbm = None

# 서울시 자치구 리스트
SEOUL_GU = [
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구",
    "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구",
    "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"
]

HOLIDAY_LIST = ['2026-01-01', '2026-02-16', '2026-02-17', '2026-02-18', '2026-03-01', '2026-03-02', '2026-05-05', '2026-05-25', '2026-06-06', '2026-08-15',
                '2026-08-17', '2026-09-24', '2026-09-25', '2026-09-26', '2026-10-03', '2026-10-05', '2026-10-09', '2026-12-25']

def prepare_features_env(gu, dong, region_code, min_temp, ground_temp, humidity, pressure, sunshine, is_holiday, lag_1d, lag_7d):
    """
    환경 변수 기반 피처 준비
    실제 모델이 필요로 하는 피처에 맞춰 조정 필요
    """
    # 지역 코드를 숫자로 변환 (모델 입력용)
    region_numeric = int(region_code) if region_code else 0
    
    # 기본 피처 구성 (모델에 맞게 수정 필요)
    features = {
        'region_code': region_numeric,
        'gu': gu,
        'dong': dong,
        'min_temp': min_temp,
        'ground_temp': ground_temp,
        'humidity': humidity,
        'pressure': pressure,
        'sunshine': sunshine,
        'is_holiday': is_holiday,
        'lag_1d': lag_1d,
        'lag_7d': lag_7d,
    }
    
    return features

def get_weather_data(date):
    date = pd.to_datetime(date)
    TMN = 100
    REH = 0
    count = 0
    print(item_list)
    for item in item_list:
        category = item.select_one('category').text
        if ((category == 'TMP') & (date == pd.to_datetime(item.select_one('fcstDate').text))):
            if TMN > float(item.fcstValue.text):
                TMN = float(item.fcstValue.text)
        if ((category == 'REH') & ((date == pd.to_datetime(item.select_one('fcstDate').text)))):
            if item.fcstValue.text:
                count+=1
                REH += float(item.fcstValue.text)
        print(item)
    
    if count == 0:
        count +=1
    return TMN, REH/count

def prepare_features_date(gu, dong, region_code, date_str):
    """
    날짜 기반 피처 준비
    날짜에서 추출할 수 있는 시간 피처들
    """
    min_temp, humidity = get_weather_data(date_str)

    # 지역 코드를 숫자로 변환
    region_numeric = int(region_code) if region_code else 0
    datetime_obj = datetime.strptime(date_str, '%Y-%m-%d')
    is_holiday = (datetime_obj.weekday()==5) or (datetime_obj.weekday()==6) or (date_str in HOLIDAY_LIST)

    features = {
        'region_code': region_numeric,
        'gu': gu,
        'dong': dong,
        'min_temp': min_temp,
        'humidity': humidity,
        'is_holiday': is_holiday,  # 0=월요일, 6=일요일
    }
    
    return features

def predict_with_model(features, mode='env'):
    """
    모델을 사용한 예측
    mode: 'env' (환경변수) 또는 'date' (날짜)
    """
    try:
        # 실제 모델의 입력 형식에 맞게 데이터 변환
        # 여기서는 예시로 구현하며, 실제 모델 구조에 맞춰 수정 필요
        
        if mode == 'env' and model_lgbm is not None:
            # LGBM 모델 사용 (환경 변수 예측)
            # 모델이 기대하는 피처 순서대로 배열 생성
            input_data = {
                '지역코드': features.get('region_code', 0),
                '최저기온(°C)': features.get('min_temp', 20),
                '0.5m 지중온도(°C)': features.get('ground_temp', 18),
                '평균 증기압(hPa)': features.get('pressure', 1013),
                '가조시간(hr)': features.get('sunshine', 14),
                '평균 상대습도(%)': features.get('humidity', 60),
                'lag_1d': features.get('lag_1d', 0),
                'lag_7d': features.get('lag_7d', 0),
                '휴일여부': features.get('is_holiday', 0),  # 0: 평일, 1: 휴일
            }
            df = pd.DataFrame([input_data])
            cat_cols = ['지역코드']
            for c in cat_cols:
                df[c] = df[c].astype("category")


            prediction = model_lgbm.predict(df)[0]
            
        elif mode == 'date' and model_forecast is not None:
            # 예보 모델 사용 (날짜 기반)
            if (features.get('min_temp', 100) == 100):
                raise Exception('예보 가능한 날짜가 아닙니다. (오늘로부터 4일이내의 날짜만 예보가 가능합니다)')
            input_data = {
                '지역코드': features.get('region_code', 0),
                '최저기온(°C)': features.get('min_temp', 20),
                '평균 상대습도(%)': features.get('humidity', 60),
                '휴일여부': features.get('is_holiday', 0),  # 0: 평일, 1: 휴일
            }
            df = pd.DataFrame([input_data])
            cat_cols = ['지역코드', '휴일여부']
            for c in cat_cols:
                df[c] = df[c].astype("category")
            
            prediction = model_forecast.predict(df)[0]
            
        else:
            # 모델이 없는 경우 더미 예측값 반환
            # 지역 코드 기반으로 기본값 설정
            region_code = features.get('region_code', 0)
            base = 60000 + (region_code % 10000)  # 지역 코드에 따라 변동
            
            if mode == 'env':
                # 환경 변수 기반 조정
                prediction = base
                prediction += (features.get('min_temp', 20) - 20) * 600
                prediction += (features.get('ground_temp', 18) - 18) * 220
                prediction += (features.get('humidity', 60) - 60) * 35
                prediction += (features.get('pressure', 1013) - 1013) * 55
                prediction += (features.get('sunshine', 14) - 14) * 180
                
                if features.get('is_holiday', 0) == 1:
                    prediction *= 0.82
            else:
                # 날짜 기반 조정
                prediction = base
                month = features.get('month', 1)
                if month in [7, 8]:
                    prediction += 4200
                if month in [12, 1, 2]:
                    prediction += 2600
            
            # 랜덤 노이즈 추가
            noise = np.random.uniform(-1500, 1500)
            prediction += noise
        
        return max(0, prediction)  # 음수 방지
        
    except Exception as e:
        print(f"예측 오류: {e}")
        import traceback
        traceback.print_exc()
        raise Exception(e)  # 기본값 반환

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    """예측 API 엔드포인트"""
    try:
        data = request.json
        mode = data.get('mode', 'env')  # 'env' 또는 'date'
        
        # 지역 정보 가져오기
        gu = data.get('gu', '강남구')
        dong = data.get('dong', '')
        region_code = int(data.get('region_code', None))
        # 로그 출력
        print(f"예측 요청 - 구: {gu}, 동: {dong}, 코드: {region_code}, 모드: {mode}")
        
        if mode == 'env':
            # 환경 변수 기반 예측
            min_temp = float(data.get('min_temp', 20))
            ground_temp = float(data.get('ground_temp', 18))
            humidity = float(data.get('humidity', 60))
            pressure = float(data.get('pressure', 1013))
            sunshine = float(data.get('sunshine', 14))
            is_holiday = int(data.get('is_holiday', 0))
            lag_1d = float(data.get('lag_1d', 0))
            lag_7d = float(data.get('lag_7d', 0))

            
            features = prepare_features_env(
                gu, dong, region_code, min_temp, ground_temp, humidity, 
                pressure, sunshine, is_holiday, lag_1d, lag_7d
            )

            prediction = predict_with_model(features, mode='env')
            
            if is_holiday == 1:
                description = "휴일 가정(수요 감소) 반영 완료"
            else:
                description = f"{gu} {dong} 환경 변수 시뮬레이션 결과"
                
            tag = "FEATURE-DRIVEN AI"
            
        else:
            # 날짜 기반 예측
            date_str = data.get('date', datetime.now().strftime('%Y-%m-%d'))
            
            features = prepare_features_date(gu, dong, region_code, date_str)
            prediction = predict_with_model(features, mode='date')
            
            # 보조 지표 계산
            description = f"{gu} {dong} 날짜별 평균 기상 데이터 분석 완료"
            tag = "DATE-BASED AI"
        
        # 응답 생성
        response = {
            'success': True,
            'prediction': int(prediction),
            'description': description,
            'tag': tag,
            'gu': gu,
            'dong': dong,
            'region_code': region_code,
            'mode': mode
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"예측 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health')
def health():
    """헬스 체크"""
    return jsonify({
        'status': 'healthy',
        'models': {
            '예보모델': model_forecast is not None,
            'LGBM': model_lgbm is not None
        }
    })

@app.route('/preprocessing')
def preprocessing():
    return render_template('preprocessing.html')

@app.route('/eda')
def eda():
    return render_template('eda.html')

if __name__ == '__main__':
    print("\n" + "="*50)
    print("전력 수요 예측 Flask 서버 시작")
    print("="*50)
    print(f"예보모델 상태: {'✓ 로드됨' if model_forecast else '✗ 미로드'}")
    print(f"LGBM 모델 상태: {'✓ 로드됨' if model_lgbm else '✗ 미로드'}")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
