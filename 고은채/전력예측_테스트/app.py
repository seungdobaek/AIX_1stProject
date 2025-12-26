from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# 모델 로드
MODEL_PATH = 'models/model.pkl'
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ 모델 로드 성공!")
except FileNotFoundError:
    print("⚠️ 모델 파일을 찾을 수 없습니다. models/model.pkl 경로를 확인하세요.")
    model = None
except Exception as e:
    print(f"❌ 모델 로드 실패: {str(e)}")
    model = None


def preprocess_input(data):
    """
    사용자 입력을 모델이 이해할 수 있는 형태로 전처리
    
    Parameters:
    -----------
    data : dict
        사용자 입력 데이터 (7개 기상 변수)
    
    Returns:
    --------
    pd.DataFrame
        모델 입력 형식으로 변환된 데이터
    """
    try:
        features = {
            '최저기온(°C)': float(data.get('temperature', 0)),
            '3.0m 지중온도(°C)': float(data.get('ground_temp', 0)),
            '합계 소형증발량(mm)': float(data.get('precipitation', 0)),
            '풍속(100m)': float(data.get('wind_speed', 0)),
            '평균 상대습도(%)': float(data.get('humidity', 0)),
            '평균 현지기압(hPa)': float(data.get('pressure', 1013.25)),
            '가조시간(hr)': float(data.get('sunshine', 0))
        }
        
        df = pd.DataFrame([features])
        return df
        
    except Exception as e:
        print(f"❌ 전처리 중 오류 발생: {str(e)}")
        raise ValueError(f"데이터 전처리 실패: {str(e)}")


@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/health')
def health():
    """
    헬스 체크 엔드포인트
    서버 상태 및 모델 로드 여부 확인
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    전력 수요 예측 API
    
    Request Body (JSON):
    {
        "temperature": float,      # 최저기온 (°C)
        "ground_temp": float,      # 지중온도 (°C)
        "precipitation": float,    # 소형증발량 (mm)
        "wind_speed": float,       # 풍속 (m/s)
        "humidity": float,         # 상대습도 (%)
        "pressure": float,         # 현지기압 (hPa)
        "sunshine": float          # 가조시간 (hr)
    }
    
    Response (JSON):
    {
        "status": "success",
        "prediction": float,       # 예측 전력량 (kWh)
        "unit": "kWh"
    }
    """
    try:
        # 1. 모델 확인
        if model is None:
            return jsonify({
                'status': 'error',
                'message': '모델이 로드되지 않았습니다.'
            }), 500
        
        # 2. 요청 데이터 받기
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': '요청 데이터가 없습니다.'
            }), 400
        
        # 3. 데이터 전처리
        features = preprocess_input(data)
        
        # 4. 예측 수행
        prediction = model.predict(features)
        prediction_value = float(prediction[0])
        
        # 5. 응답 반환
        return jsonify({
            'status': 'success',
            'prediction': round(prediction_value, 2),
            'unit': 'kWh'
        }), 200
        
    except ValueError as ve:
        return jsonify({
            'status': 'error',
            'message': str(ve)
        }), 400
        
    except Exception as e:
        print(f"❌ 예측 중 오류: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'서버 오류가 발생했습니다: {str(e)}'
        }), 500


@app.errorhandler(404)
def not_found(error):
    """404 에러 핸들러"""
    return jsonify({
        'status': 'error',
        'message': '페이지를 찾을 수 없습니다.'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500 에러 핸들러"""
    return jsonify({
        'status': 'error',
        'message': '서버 내부 오류가 발생했습니다.'
    }), 500


if __name__ == '__main__':
    # 개발 모드로 실행
    # 프로덕션 환경에서는 gunicorn 등 사용 권장
    app.run(debug=True, host='0.0.0.0', port=5000)
