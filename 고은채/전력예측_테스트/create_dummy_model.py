"""
테스트용 더미 모델 생성 스크립트
팀원에게 실제 모델을 받기 전까지 임시로 사용
"""
import pickle
import xgboost as xgb
import numpy as np
import os

# models 폴더 확인 및 생성
if not os.path.exists('models'):
    os.makedirs('models')
    print("📁 models 폴더 생성")

# 더미 모델 생성
print("🔧 더미 모델 생성 중...")
model = xgb.XGBRegressor(n_estimators=10, max_depth=3, random_state=42)

# 더미 데이터로 학습
X_dummy = np.random.rand(100, 7)
y_dummy = np.random.rand(100) * 1000000

print("🎓 모델 학습 중...")
model.fit(X_dummy, y_dummy, verbose=False)

# 모델 저장
print("💾 모델 저장 중...")
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ 더미 모델 생성 완료!")
print("📍 저장 위치: models/model.pkl")
print("\n⚠️  이것은 테스트용 더미 모델입니다.")
print("⚠️  실제 프로젝트에는 팀원이 학습한 모델을 사용하세요!")
