"""
변수 선택 및 EDA (탐색적 데이터 분석)
====================================

목표: 훈련에 사용할 변수 10개 선택

데이터: 최종_통합_데이터_완벽.csv
분석 항목:
1. 기본 데이터 탐색
2. 결측치 분석
3. 분포 분석
4. 상관관계 분석
5. 변수 중요도 분석
6. 최종 변수 10개 추천

필요한 패키지:
pip install pandas numpy matplotlib seaborn scipy scikit-learn

사용법:
python 변수선택_EDA.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# 메인 프로그램
# ============================================================================

def main():
    """메인 실행 함수"""
    
    print("="*100)
    print("변수 선택 및 EDA (Exploratory Data Analysis)")
    print("="*100)
    print("\n목표: 훈련에 사용할 변수 10개 선택")
    
    data_file = '최종_통합_데이터_완벽.csv'
    
    if not os.path.exists(data_file):
        print(f"\n❌ '{data_file}' 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n✓ 데이터 파일 발견: {data_file}")
    
    # 메뉴
    while True:
        print("\n" + "="*100)
        print("분석 단계 선택:")
        print("="*100)
        print("\n1. 📊 1단계: 기본 데이터 탐색")
        print("2. 🔍 2단계: 결측치 및 이상치 분석")
        print("3. 📈 3단계: 분포 분석")
        print("4. 🔗 4단계: 상관관계 분석")
        print("5. 🎯 5단계: 변수 중요도 분석")
        print("6. ✅ 6단계: 최종 변수 10개 추천")
        print("7. 📋 전체 리포트 생성")
        print("0. ❌ 종료")
        
        choice = input("\n선택: ").strip()
        
        if choice == '1':
            step1_basic_exploration(data_file)
        elif choice == '2':
            step2_missing_outliers(data_file)
        elif choice == '3':
            step3_distribution(data_file)
        elif choice == '4':
            step4_correlation(data_file)
        elif choice == '5':
            step5_feature_importance(data_file)
        elif choice == '6':
            step6_final_selection(data_file)
        elif choice == '7':
            generate_full_report(data_file)
        elif choice == '0':
            print("\n종료합니다.")
            break
        else:
            print("\n❌ 잘못된 선택입니다.")


# ============================================================================
# 1단계: 기본 데이터 탐색
# ============================================================================

def step1_basic_exploration(data_file):
    """1단계: 기본 데이터 탐색"""
    print("\n" + "="*100)
    print("📊 1단계: 기본 데이터 탐색")
    print("="*100)
    
    df = pd.read_csv(data_file)
    df['일자'] = pd.to_datetime(df['일자'])
    
    print(f"\n[1] 데이터 기본 정보")
    print(f"  - 행 수: {len(df):,}")
    print(f"  - 열 수: {len(df.columns)}")
    print(f"  - 기간: {df['일자'].min().date()} ~ {df['일자'].max().date()}")
    print(f"  - 총 일수: {(df['일자'].max() - df['일자'].min()).days + 1}일")
    
    print(f"\n[2] 컬럼 목록 및 데이터 타입")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_pct = (df[col].isna().sum() / len(df)) * 100
        print(f"  {i:2d}. {col:30s} | {str(dtype):10s} | Non-null: {non_null:6,} ({100-null_pct:5.1f}%)")
    
    # 타겟 변수
    target = '일별_전력사용량_KWH'
    
    print(f"\n[3] 타겟 변수: {target}")
    print(f"  - 평균: {df[target].mean():,.0f} kWh")
    print(f"  - 중앙값: {df[target].median():,.0f} kWh")
    print(f"  - 표준편차: {df[target].std():,.0f} kWh")
    print(f"  - 최소: {df[target].min():,.0f} kWh")
    print(f"  - 최대: {df[target].max():,.0f} kWh")
    print(f"  - 변동계수(CV): {(df[target].std() / df[target].mean()):.2f}")
    
    # 수치형 변수와 범주형 변수 구분
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 제외할 컬럼
    exclude_cols = ['BJDONG_CD', '법정동코드', '시군구코드', '법정동_세부코드']
    numeric_features = [col for col in numeric_cols if col not in exclude_cols and col != target]
    
    print(f"\n[4] 변수 타입 분류")
    print(f"  - 수치형 변수: {len(numeric_features)}개")
    print(f"  - 범주형 변수: {len(categorical_cols)}개")
    
    print(f"\n[5] 수치형 변수 목록 ({len(numeric_features)}개)")
    for i, col in enumerate(numeric_features, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n[6] 범주형 변수 목록 ({len(categorical_cols)}개)")
    for i, col in enumerate(categorical_cols, 1):
        unique_count = df[col].nunique()
        print(f"  {i:2d}. {col:20s} - {unique_count}개 고유값")
    
    print("\n✓ 1단계 완료!")


# ============================================================================
# 2단계: 결측치 및 이상치 분석
# ============================================================================

def step2_missing_outliers(data_file):
    """2단계: 결측치 및 이상치 분석"""
    print("\n" + "="*100)
    print("🔍 2단계: 결측치 및 이상치 분석")
    print("="*100)
    
    df = pd.read_csv(data_file)
    df['일자'] = pd.to_datetime(df['일자'])
    
    # 결측치 분석
    print("\n[1] 결측치 분석")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        '컬럼명': missing.index,
        '결측치수': missing.values,
        '비율(%)': missing_pct.values
    })
    missing_df = missing_df[missing_df['결측치수'] > 0].sort_values('결측치수', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("  ✓ 결측치 없음!")
    
    # 이상치 분석 (수치형 변수)
    target = '일별_전력사용량_KWH'
    exclude_cols = ['BJDONG_CD', '법정동코드', '시군구코드', '법정동_세부코드']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"\n[2] 이상치 분석 (IQR 방법)")
    print(f"  기준: Q1 - 1.5*IQR 미만 또는 Q3 + 1.5*IQR 초과")
    
    outlier_summary = []
    for col in numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        
        outlier_summary.append({
            '변수명': col,
            '이상치수': outlier_count,
            '비율(%)': outlier_pct,
            '하한': lower_bound,
            '상한': upper_bound
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    outlier_df = outlier_df[outlier_df['이상치수'] > 0].sort_values('이상치수', ascending=False)
    
    if len(outlier_df) > 0:
        print("\n  이상치가 있는 변수:")
        for _, row in outlier_df.head(10).iterrows():
            print(f"    {row['변수명']:30s}: {row['이상치수']:5d}개 ({row['비율(%)']:5.1f}%)")
    else:
        print("  ✓ 이상치 없음!")
    
    # 타겟 변수 이상치 상세 분석
    print(f"\n[3] 타겟 변수 이상치 상세")
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = df[(df[target] < lower) | (df[target] > upper)]
    print(f"  - 이상치 개수: {len(outliers):,}개 ({len(outliers)/len(df)*100:.1f}%)")
    print(f"  - 하한: {lower:,.0f} kWh")
    print(f"  - 상한: {upper:,.0f} kWh")
    
    if len(outliers) > 0:
        print(f"\n  상위 5개 이상치:")
        top_outliers = df.nlargest(5, target)[['일자', '법정동명', target, '평균기온(°C)']]
        print(top_outliers.to_string(index=False))
    
    print("\n✓ 2단계 완료!")


# ============================================================================
# 3단계: 분포 분석
# ============================================================================

def step3_distribution(data_file):
    """3단계: 분포 분석"""
    print("\n" + "="*100)
    print("📈 3단계: 분포 분석")
    print("="*100)
    
    df = pd.read_csv(data_file)
    df['일자'] = pd.to_datetime(df['일자'])
    
    target = '일별_전력사용량_KWH'
    exclude_cols = ['BJDONG_CD', '법정동코드', '시군구코드', '법정동_세부코드']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_cols if col not in exclude_cols and col != target]
    
    # 정규성 검정
    print("\n[1] 정규성 검정 (Shapiro-Wilk Test)")
    print("  기준: p-value > 0.05 → 정규분포")
    
    # 샘플링 (전체 데이터는 너무 크므로)
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    normality_results = []
    for col in numeric_features:
        if df_sample[col].notna().sum() > 3:  # 최소 3개 이상
            stat, p_value = stats.shapiro(df_sample[col].dropna())
            is_normal = "정규분포" if p_value > 0.05 else "비정규"
            normality_results.append({
                '변수명': col,
                'p-value': p_value,
                '판정': is_normal
            })
    
    norm_df = pd.DataFrame(normality_results).sort_values('p-value', ascending=False)
    print("\n  정규성 검정 결과 (상위 15개):")
    for _, row in norm_df.head(15).iterrows():
        print(f"    {row['변수명']:30s}: p={row['p-value']:.4f} ({row['판정']})")
    
    # 왜도 및 첨도
    print("\n[2] 왜도(Skewness) 및 첨도(Kurtosis)")
    print("  왜도: 0에 가까울수록 대칭, |왜도| < 1 적절")
    print("  첨도: 0에 가까울수록 정규분포, |첨도| < 3 적절")
    
    skew_kurt = []
    for col in numeric_features:
        skewness = df[col].skew()
        kurtosis = df[col].kurtosis()
        skew_kurt.append({
            '변수명': col,
            '왜도': skewness,
            '첨도': kurtosis,
            '왜도판정': '적절' if abs(skewness) < 1 else '치우침',
            '첨도판정': '적절' if abs(kurtosis) < 3 else '뾰족/평평'
        })
    
    sk_df = pd.DataFrame(skew_kurt)
    print("\n  왜도가 적절한 변수:")
    appropriate = sk_df[sk_df['왜도판정'] == '적절'].sort_values('왜도', key=abs)
    for _, row in appropriate.head(10).iterrows():
        print(f"    {row['변수명']:30s}: 왜도={row['왜도']:6.2f}, 첨도={row['첨도']:6.2f}")
    
    # 시각화
    print("\n[3] 분포 시각화 생성 중...")
    output_dir = 'eda_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 주요 변수 4개 분포
    key_vars = ['평균기온(°C)', '최저기온(°C)', '최고기온(°C)', '일강수량(mm)']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, var in enumerate(key_vars):
        axes[i].hist(df[var], bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel(var, fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].set_title(f'Distribution of {var}', fontsize=11, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        # 통계량 표시
        mean = df[var].mean()
        median = df[var].median()
        axes[i].axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.1f}')
        axes[i].axvline(median, color='blue', linestyle='--', linewidth=2, label=f'Median: {median:.1f}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_distribution_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 저장: {output_dir}/3_distribution_analysis.png")
    
    print("\n✓ 3단계 완료!")


# ============================================================================
# 4단계: 상관관계 분석
# ============================================================================

def step4_correlation(data_file):
    """4단계: 상관관계 분석"""
    print("\n" + "="*100)
    print("🔗 4단계: 상관관계 분석")
    print("="*100)
    
    df = pd.read_csv(data_file)
    df['일자'] = pd.to_datetime(df['일자'])
    
    target = '일별_전력사용량_KWH'
    exclude_cols = ['BJDONG_CD', '법정동코드', '시군구코드', '법정동_세부코드']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_cols if col not in exclude_cols]
    
    # 상관계수 계산
    corr_matrix = df[numeric_features].corr()
    
    # 타겟 변수와의 상관관계
    print("\n[1] 타겟 변수와의 상관관계 (Pearson)")
    target_corr = corr_matrix[target].drop(target).sort_values(ascending=False, key=abs)
    
    print("\n  상관계수가 높은 변수 (Top 15):")
    for var, corr in target_corr.head(15).items():
        strength = "매우강함" if abs(corr) >= 0.7 else "강함" if abs(corr) >= 0.5 else "중간" if abs(corr) >= 0.3 else "약함"
        print(f"    {var:30s}: {corr:7.4f} ({strength})")
    
    print("\n  상관계수가 낮은 변수 (Bottom 5):")
    for var, corr in target_corr.tail(5).items():
        print(f"    {var:30s}: {corr:7.4f}")
    
    # 다중공선성 검사
    print("\n[2] 다중공선성 검사 (변수 간 상관관계)")
    print("  기준: |상관계수| > 0.8 → 다중공선성 의심")
    
    high_corr_pairs = []
    for i in range(len(numeric_features)):
        for j in range(i+1, len(numeric_features)):
            var1 = numeric_features[i]
            var2 = numeric_features[j]
            if var1 in corr_matrix.columns and var2 in corr_matrix.columns:
                corr_val = corr_matrix.loc[var1, var2]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append({
                        '변수1': var1,
                        '변수2': var2,
                        '상관계수': corr_val
                    })
    
    if high_corr_pairs:
        print(f"\n  높은 상관관계를 가진 변수 쌍 ({len(high_corr_pairs)}개):")
        for pair in sorted(high_corr_pairs, key=lambda x: abs(x['상관계수']), reverse=True)[:10]:
            print(f"    {pair['변수1']:25s} - {pair['변수2']:25s}: {pair['상관계수']:6.3f}")
    else:
        print("  ✓ 다중공선성 문제 없음!")
    
    # 시각화
    print("\n[3] 상관관계 히트맵 생성 중...")
    output_dir = 'eda_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 주요 변수만 선택 (상위 12개)
    top_vars = target_corr.head(12).index.tolist()
    top_vars.insert(0, target)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(df[top_vars].corr(), annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap (Top 12 Variables + Target)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 저장: {output_dir}/4_correlation_heatmap.png")
    
    # 타겟과의 산점도
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    top_9_vars = target_corr.head(9).index.tolist()
    
    for i, var in enumerate(top_9_vars):
        axes[i].scatter(df[var], df[target]/1000000, alpha=0.3, s=5)
        axes[i].set_xlabel(var, fontsize=9)
        axes[i].set_ylabel('Power (GWh)', fontsize=9)
        corr = target_corr[var]
        axes[i].set_title(f'{var}\n(r = {corr:.3f})', fontsize=10, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Scatter Plots: Top 9 Correlated Variables vs Target', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_scatter_top9.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 저장: {output_dir}/4_scatter_top9.png")
    
    print("\n✓ 4단계 완료!")


# ============================================================================
# 5단계: 변수 중요도 분석
# ============================================================================

def step5_feature_importance(data_file):
    """5단계: 변수 중요도 분석"""
    print("\n" + "="*100)
    print("🎯 5단계: 변수 중요도 분석 (Random Forest)")
    print("="*100)
    
    df = pd.read_csv(data_file)
    df['일자'] = pd.to_datetime(df['일자'])
    
    target = '일별_전력사용량_KWH'
    exclude_cols = ['BJDONG_CD', '법정동코드', '시군구코드', '법정동_세부코드']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_cols if col not in exclude_cols and col != target]
    
    print("\n[1] Random Forest 모델 학습 중...")
    
    # 결측치 제거
    df_clean = df[numeric_features + [target]].dropna()
    
    X = df_clean[numeric_features]
    y = df_clean[target]
    
    # 모델 학습
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X, y)
    
    # 변수 중요도
    importances = rf.feature_importances_
    feature_importance = pd.DataFrame({
        '변수명': numeric_features,
        '중요도': importances
    }).sort_values('중요도', ascending=False)
    
    print("\n[2] 변수 중요도 순위 (Top 20)")
    for i, row in feature_importance.head(20).iterrows():
        print(f"  {i+1:2d}. {row['변수명']:30s}: {row['중요도']:.4f}")
    
    # 시각화
    print("\n[3] 변수 중요도 시각화 생성 중...")
    output_dir = 'eda_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(12, 10))
    top_20 = feature_importance.head(20)
    colors_grad = plt.cm.viridis(np.linspace(0, 1, 20))
    
    plt.barh(range(len(top_20)), top_20['중요도'].values, color=colors_grad)
    plt.yticks(range(len(top_20)), top_20['변수명'].values)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 20 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # 값 표시
    for i, (idx, row) in enumerate(top_20.iterrows()):
        plt.text(row['중요도'], i, f" {row['중요도']:.4f}", 
                 va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/5_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 저장: {output_dir}/5_feature_importance.png")
    
    print("\n✓ 5단계 완료!")
    
    return feature_importance


# ============================================================================
# 6단계: 최종 변수 10개 추천
# ============================================================================

def step6_final_selection(data_file):
    """6단계: 최종 변수 10개 추천"""
    print("\n" + "="*100)
    print("✅ 6단계: 최종 변수 10개 선택")
    print("="*100)
    
    df = pd.read_csv(data_file)
    df['일자'] = pd.to_datetime(df['일자'])
    
    target = '일별_전력사용량_KWH'
    exclude_cols = ['BJDONG_CD', '법정동코드', '시군구코드', '법정동_세부코드']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_cols if col not in exclude_cols and col != target]
    
    # 1. 상관계수
    print("\n[1] 상관계수 기반 추천")
    corr_matrix = df[numeric_features + [target]].corr()
    target_corr = corr_matrix[target].drop(target).sort_values(ascending=False, key=abs)
    corr_top10 = target_corr.head(10)
    
    print("  상관계수 Top 10:")
    for i, (var, corr) in enumerate(corr_top10.items(), 1):
        print(f"    {i:2d}. {var:30s}: {corr:7.4f}")
    
    # 2. Random Forest 중요도
    print("\n[2] Random Forest 중요도 기반 추천")
    df_clean = df[numeric_features + [target]].dropna()
    X = df_clean[numeric_features]
    y = df_clean[target]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X, y)
    
    importances = pd.DataFrame({
        '변수명': numeric_features,
        '중요도': rf.feature_importances_
    }).sort_values('중요도', ascending=False)
    
    rf_top10 = importances.head(10)
    
    print("  중요도 Top 10:")
    for i, row in rf_top10.iterrows():
        print(f"    {i+1:2d}. {row['변수명']:30s}: {row['중요도']:.4f}")
    
    # 3. 종합 점수 (상관계수 + 중요도)
    print("\n[3] 종합 점수 기반 최종 추천")
    
    # 정규화
    corr_norm = (abs(target_corr) - abs(target_corr).min()) / (abs(target_corr).max() - abs(target_corr).min())
    imp_norm = (importances.set_index('변수명')['중요도'] - importances['중요도'].min()) / \
               (importances['중요도'].max() - importances['중요도'].min())
    
    # 종합 점수 (가중평균: 상관계수 40%, 중요도 60%)
    combined_score = {}
    for var in numeric_features:
        corr_score = corr_norm.get(var, 0) * 0.4
        imp_score = imp_norm.get(var, 0) * 0.6
        combined_score[var] = corr_score + imp_score
    
    combined_df = pd.DataFrame(list(combined_score.items()), 
                                columns=['변수명', '종합점수']).sort_values('종합점수', ascending=False)
    
    final_top10 = combined_df.head(10)
    
    print("\n  ★ 최종 추천 변수 10개 ★")
    for i, row in final_top10.iterrows():
        var = row['변수명']
        score = row['종합점수']
        corr = target_corr.get(var, 0)
        imp = importances[importances['변수명'] == var]['중요도'].values[0] if var in importances['변수명'].values else 0
        
        print(f"    {i+1:2d}. {var:30s}")
        print(f"        - 종합점수: {score:.4f}")
        print(f"        - 상관계수: {corr:7.4f}")
        print(f"        - 중요도:   {imp:.4f}")
    
    # 저장
    output_dir = 'eda_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # CSV 저장
    final_selection = pd.DataFrame({
        '순위': range(1, 11),
        '변수명': final_top10['변수명'].values,
        '종합점수': final_top10['종합점수'].values,
        '상관계수': [target_corr.get(var, 0) for var in final_top10['변수명']],
        '중요도': [importances[importances['변수명'] == var]['중요도'].values[0] 
                 if var in importances['변수명'].values else 0 
                 for var in final_top10['변수명']]
    })
    
    final_selection.to_csv(f'{output_dir}/최종선택_변수10개.csv', index=False, encoding='utf-8-sig')
    print(f"\n  ✓ 저장: {output_dir}/최종선택_변수10개.csv")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 상관계수 vs 중요도
    axes[0].scatter([corr_norm.get(var, 0) for var in final_top10['변수명']],
                    [imp_norm.get(var, 0) for var in final_top10['변수명']],
                    s=200, alpha=0.6, c=range(10), cmap='viridis')
    
    for i, var in enumerate(final_top10['변수명']):
        axes[0].annotate(f"{i+1}", 
                        (corr_norm.get(var, 0), imp_norm.get(var, 0)),
                        ha='center', va='center', fontweight='bold')
    
    axes[0].set_xlabel('Normalized Correlation (40%)', fontsize=11)
    axes[0].set_ylabel('Normalized Importance (60%)', fontsize=11)
    axes[0].set_title('Feature Selection: Correlation vs Importance', 
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 종합 점수 바 차트
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    axes[1].barh(range(10), final_top10['종합점수'].values, color=colors)
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels([f"{i+1}. {var}" for i, var in enumerate(final_top10['변수명'])])
    axes[1].set_xlabel('Combined Score', fontsize=11)
    axes[1].set_title('Final Top 10 Variables', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    for i, score in enumerate(final_top10['종합점수'].values):
        axes[1].text(score, i, f' {score:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/6_final_selection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 저장: {output_dir}/6_final_selection.png")
    
    print("\n✓ 6단계 완료!")
    print("\n" + "="*100)
    print("🎉 변수 선택 완료! 최종 10개 변수를 확인하세요.")
    print("="*100)


# ============================================================================
# 7. 전체 리포트 생성
# ============================================================================

def generate_full_report(data_file):
    """전체 리포트 생성"""
    print("\n" + "="*100)
    print("📋 전체 EDA 리포트 생성")
    print("="*100)
    print("\n모든 단계를 순차적으로 실행합니다...")
    
    input("\n1단계 시작 - Enter를 누르세요...")
    step1_basic_exploration(data_file)
    
    input("\n2단계 시작 - Enter를 누르세요...")
    step2_missing_outliers(data_file)
    
    input("\n3단계 시작 - Enter를 누르세요...")
    step3_distribution(data_file)
    
    input("\n4단계 시작 - Enter를 누르세요...")
    step4_correlation(data_file)
    
    input("\n5단계 시작 - Enter를 누르세요...")
    step5_feature_importance(data_file)
    
    input("\n6단계 시작 - Enter를 누르세요...")
    step6_final_selection(data_file)
    
    print("\n" + "="*100)
    print("✅ 전체 리포트 생성 완료!")
    print("="*100)
    print("\n생성된 파일:")
    print("  - eda_results/3_distribution_analysis.png")
    print("  - eda_results/4_correlation_heatmap.png")
    print("  - eda_results/4_scatter_top9.png")
    print("  - eda_results/5_feature_importance.png")
    print("  - eda_results/6_final_selection.png")
    print("  - eda_results/최종선택_변수10개.csv")


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    main()
