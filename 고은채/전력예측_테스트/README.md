# 🔌 서울시 전력 수요 예측 Flask API

XGBoost 모델을 활용한 서울시 전력 수요 예측 웹 애플리케이션

## 📁 프로젝트 구조

```
project/
├── app.py                      # 메인 Flask 애플리케이션
├── config.py                   # 설정 관리
├── test_api.py                 # API 테스트 스크립트
├── requirements.txt            # Python 패키지 목록
├── .env.example               # 환경 변수 예시
├── README.md                   # 이 파일
│
├── models/                     # 학습된 모델 저장
│   └── model.pkl              # XGBoost 모델 파일
│
├── templates/                  # HTML 템플릿
│   ├── index.html             # 메인 페이지
│   ├── result.html            # 예측 결과 페이지
│   └── about.html             # 프로젝트 소개
│
├── static/                     # 정적 파일
│   ├── css/
│   │   └── style.css          # 스타일시트
│   ├── js/
│   │   └── main.js            # JavaScript
│   └── images/
│       └── logo.png
│
└── uploads/                    # 업로드 파일 임시 저장
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env.example을 .env로 복사
cp .env.example .env

# .env 파일 수정 (필요시)
# SECRET_KEY, MODEL_PATH 등 설정
```

### 3. 모델 파일 준비

```bash
# models 디렉토리 생성
mkdir models

# 학습된 모델 파일(model.pkl)을 models/ 디렉토리에 복사
```

### 4. 서버 실행

```bash
# 개발 모드로 실행
python app.py

# 또는 Flask CLI 사용
export FLASK_APP=app.py
flask run

# 프로덕션 모드 (Gunicorn 사용)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

서버가 실행되면 http://localhost:5000 에서 접속 가능합니다.

## 📡 API 엔드포인트

### 1. 헬스 체크
```http
GET /health
```

**응답 예시:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-12-26T14:30:00"
}
```

### 2. 모델 정보 조회
```http
GET /api/model-info
```

**응답 예시:**
```json
{
  "status": "success",
  "model_type": "XGBoost",
  "features": [
    "최저기온(°C)",
    "3.0m 지중온도(°C)",
    "합계 소형증발량(mm)",
    "풍속(100m)",
    "평균 상대습도(%)",
    "평균 현지기압(hPa)",
    "가조시간(hr)"
  ]
}
```

### 3. 단일 예측
```http
POST /api/predict
Content-Type: application/json
```

**요청 예시:**
```json
{
  "temperature": -5.2,
  "ground_temp": 8.3,
  "precipitation": 0.0,
  "wind_speed": 3.5,
  "humidity": 65.0,
  "pressure": 1013.25,
  "sunshine": 5.5,
  "date": "2024-12-26",
  "time": "14:00"
}
```

**응답 예시:**
```json
{
  "status": "success",
  "prediction": 1234567.89,
  "unit": "kWh",
  "confidence": "high",
  "message": "예측이 완료되었습니다.",
  "input_summary": {
    "온도": "-5.2°C",
    "습도": "65.0%",
    "풍속": "3.5 m/s"
  }
}
```

### 4. 일괄 예측 (CSV)
```http
POST /api/batch-predict
Content-Type: multipart/form-data
```

**CSV 형식:**
```csv
temperature,ground_temp,precipitation,wind_speed,humidity,pressure,sunshine
-5.2,8.3,0.0,3.5,65.0,1013.25,5.5
10.5,12.0,2.5,4.0,70.0,1015.0,8.0
```

## 🧪 테스트

```bash
# API 테스트 스크립트 실행
python test_api.py

# 또는 curl로 직접 테스트
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": -5.2,
    "ground_temp": 8.3,
    "precipitation": 0.0,
    "wind_speed": 3.5,
    "humidity": 65.0,
    "pressure": 1013.25,
    "sunshine": 5.5
  }'
```

## 📊 입력 변수 범위

| 변수 | 범위 | 단위 |
|------|------|------|
| 최저기온 (temperature) | -30 ~ 40 | °C |
| 지중온도 (ground_temp) | -10 ~ 50 | °C |
| 소형증발량 (precipitation) | 0 ~ 500 | mm |
| 풍속 (wind_speed) | 0 ~ 50 | m/s |
| 상대습도 (humidity) | 0 ~ 100 | % |
| 현지기압 (pressure) | 950 ~ 1050 | hPa |
| 가조시간 (sunshine) | 0 ~ 24 | hr |

## 🔒 보안 고려사항

1. **SECRET_KEY**: `.env` 파일에서 안전한 키로 변경
2. **CORS**: 프로덕션 환경에서는 특정 도메인만 허용
3. **파일 업로드**: 파일 크기 제한 및 타입 검증
4. **Rate Limiting**: 추후 추가 권장 (Flask-Limiter 등)

## 🐛 트러블슈팅

### 모델 파일을 찾을 수 없음
```
⚠️ 모델 파일을 찾을 수 없습니다. models/model.pkl 경로를 확인하세요.
```
→ `models/model.pkl` 파일이 존재하는지 확인

### 패키지 설치 오류
```
ERROR: Could not find a version that satisfies the requirement...
```
→ Python 버전 확인 (3.8 이상 권장)
→ `pip install --upgrade pip`

### CORS 오류
```
Access to fetch at '...' from origin '...' has been blocked by CORS policy
```
→ `.env` 파일의 `CORS_ORIGINS`에 프론트엔드 URL 추가

## 📝 다음 단계

- [ ] 프론트엔드 개발 (templates/ 작성)
- [ ] 데이터베이스 연동 (예측 이력 저장)
- [ ] 사용자 인증 추가
- [ ] 실시간 모니터링 대시보드
- [ ] Docker 컨테이너화
- [ ] CI/CD 파이프라인 구축

## 👥 팀원

- 백엔드: [이름]
- 모델링: [이름]
- 프론트엔드: [이름]

## 📄 라이센스

MIT License

## 📞 문의

프로젝트 관련 문의: [이메일]
```

이제 완성된 코드를 outputs 디렉토리로 이동하겠습니다!

