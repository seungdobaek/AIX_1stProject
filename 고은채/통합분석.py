"""
전력 사용량 & 날씨 데이터 통합 분석 프로그램
============================================

필요한 패키지:
pip install pandas numpy matplotlib seaborn scipy

사용법:
python 통합분석.py

작성일: 2024-12-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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
    print("전력 사용량 & 날씨 데이터 통합 분석 프로그램")
    print("="*100)
    
    # 데이터 파일 경로
    data_file = '전력사용량_날씨_통합데이터.csv'
    
    # 파일 존재 확인
    if not os.path.exists(data_file):
        print(f"\n❌ 오류: '{data_file}' 파일을 찾을 수 없습니다.")
        print("\n현재 디렉토리에 데이터 파일을 넣어주세요.")
        return
    
    print(f"\n✓ 데이터 파일 발견: {data_file}")
    
    # 메뉴
    while True:
        print("\n" + "="*100)
        print("실행할 작업을 선택하세요:")
        print("="*100)
        print("\n1. 📊 기본 통계 분석")
        print("2. 📈 상세 분석 (월별/계절별/요일별)")
        print("3. 🎨 시각화 생성 (그래프 8개)")
        print("4. 🔍 맞춤 분석")
        print("5. ⚡ 전체 실행 (분석 + 시각화)")
        print("0. ❌ 종료")
        
        choice = input("\n선택 (0-5): ").strip()
        
        if choice == '1':
            basic_analysis(data_file)
        elif choice == '2':
            detailed_analysis(data_file)
        elif choice == '3':
            visualization(data_file)
        elif choice == '4':
            custom_analysis(data_file)
        elif choice == '5':
            run_all(data_file)
        elif choice == '0':
            print("\n프로그램을 종료합니다.")
            break
        else:
            print("\n❌ 잘못된 선택입니다.")


# ============================================================================
# 1. 기본 통계 분석
# ============================================================================

def basic_analysis(data_file):
    """기본 통계 분석"""
    print("\n" + "="*100)
    print("📊 기본 통계 분석")
    print("="*100)
    
    df = pd.read_csv(data_file)
    df['일자'] = pd.to_datetime(df['일자'])
    
    print(f"\n데이터 크기: {len(df):,}행 × {len(df.columns)}열")
    print(f"기간: {df['일자'].min().date()} ~ {df['일자'].max().date()}")
    print(f"결측치: {df.isnull().sum().sum()}개")
    
    print("\n주요 변수 기술 통계:")
    key_cols = ['일별_전력사용량_KWH', '평균기온(°C)', '최저기온(°C)', 
                '최고기온(°C)', '일강수량(mm)', '평균 풍속(m/s)', '평균 상대습도(%)']
    print(df[key_cols].describe().round(2))
    
    # 상관관계
    print("\n전력 사용량과 다른 변수들의 상관계수:")
    corr = df[key_cols].corr()['일별_전력사용량_KWH'].sort_values(ascending=False)
    print(corr.round(3))
    
    print("\n✓ 분석 완료!")


# ============================================================================
# 2. 상세 분석
# ============================================================================

def detailed_analysis(data_file):
    """상세 분석"""
    print("\n" + "="*100)
    print("📈 상세 분석")
    print("="*100)
    
    df = pd.read_csv(data_file)
    df['일자'] = pd.to_datetime(df['일자'])
    df['연도'] = df['일자'].dt.year
    df['월'] = df['일자'].dt.month
    df['요일'] = df['일자'].dt.dayofweek
    
    # 계절 구분
    def get_season(month):
        if month in [3, 4, 5]:
            return '봄'
        elif month in [6, 7, 8]:
            return '여름'
        elif month in [9, 10, 11]:
            return '가을'
        else:
            return '겨울'
    
    df['계절'] = df['월'].apply(get_season)
    
    # 냉난방도일 계산
    df['CDD'] = df['평균기온(°C)'].apply(lambda x: max(0, x - 18))
    df['HDD'] = df['평균기온(°C)'].apply(lambda x: max(0, 18 - x))
    
    # 1. 계절별 분석
    print("\n[1] 계절별 평균")
    seasonal_stats = df.groupby('계절').agg({
        '일별_전력사용량_KWH': 'mean',
        '평균기온(°C)': 'mean',
        '일강수량(mm)': 'mean'
    }).round(0)
    print(seasonal_stats)
    
    # 2. 요일별 분석
    print("\n[2] 요일별 평균")
    weekday_names = ['월', '화', '수', '목', '금', '토', '일']
    weekday_stats = df.groupby('요일')['일별_전력사용량_KWH'].mean()
    for i, usage in enumerate(weekday_stats):
        print(f"  {weekday_names[i]}요일: {usage:,.0f} kWh")
    
    # 3. 월별 추이
    print("\n[3] 월별 평균 (최근 12개월)")
    df['연월'] = df['일자'].dt.to_period('M')
    monthly_stats = df.groupby('연월').agg({
        '일별_전력사용량_KWH': 'mean',
        '평균기온(°C)': 'mean',
        '일강수량(mm)': 'sum'
    }).tail(12).round(0)
    print(monthly_stats)
    
    # 4. 극한 기상 조건
    print("\n[4] 극한 기상 조건별 평균 전력 사용량")
    hot_days = df[df['평균기온(°C)'] > 28]
    cold_days = df[df['평균기온(°C)'] < 0]
    normal_days = df[(df['평균기온(°C)'] >= 10) & (df['평균기온(°C)'] <= 25)]
    
    print(f"  폭염일 (>28°C): {hot_days['일별_전력사용량_KWH'].mean():,.0f} kWh ({len(hot_days)}일)")
    print(f"  한파일 (<0°C): {cold_days['일별_전력사용량_KWH'].mean():,.0f} kWh ({len(cold_days)}일)")
    print(f"  일반일 (10~25°C): {normal_days['일별_전력사용량_KWH'].mean():,.0f} kWh ({len(normal_days)}일)")
    
    # 5. 냉난방도일 상관관계
    print("\n[5] 냉난방도일(Degree Days) 상관관계")
    print(f"  냉방도일(CDD): {df[['CDD', '일별_전력사용량_KWH']].corr().iloc[0,1]:.3f}")
    print(f"  난방도일(HDD): {df[['HDD', '일별_전력사용량_KWH']].corr().iloc[0,1]:.3f}")
    
    # 6. 기온대별 분석
    print("\n[6] 기온대별 평균 전력 사용량")
    df['기온대'] = pd.cut(df['평균기온(°C)'], 
                        bins=[-20, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40],
                        labels=['<-10', '-10~-5', '-5~0', '0~5', '5~10', 
                               '10~15', '15~20', '20~25', '25~30', '>30'])
    temp_usage = df.groupby('기온대')['일별_전력사용량_KWH'].agg(['mean', 'count'])
    print(temp_usage.round(0))
    
    # CSV 저장
    print("\n[7] 분석 결과 저장")
    seasonal_stats.to_csv('분석결과_계절별.csv', encoding='utf-8-sig')
    monthly_stats.to_csv('분석결과_월별.csv', encoding='utf-8-sig')
    temp_usage.to_csv('분석결과_기온대별.csv', encoding='utf-8-sig')
    print("  ✓ 분석결과_계절별.csv")
    print("  ✓ 분석결과_월별.csv")
    print("  ✓ 분석결과_기온대별.csv")
    
    print("\n✓ 분석 완료!")


# ============================================================================
# 3. 시각화
# ============================================================================

def visualization(data_file):
    """시각화 생성"""
    print("\n" + "="*100)
    print("🎨 시각화 생성")
    print("="*100)
    
    df = pd.read_csv(data_file)
    df['일자'] = pd.to_datetime(df['일자'])
    
    # 출력 폴더 생성
    output_dir = 'visualizations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n그래프 생성 중...")
    colors = sns.color_palette('husl', 10)
    
    # 1. 기온 vs 전력 산점도
    print("  [1/8] 기온-전력 산점도...")
    plt.figure(figsize=(10, 6))
    plt.scatter(df['평균기온(°C)'], df['일별_전력사용량_KWH']/1000000, 
                alpha=0.3, s=10, c=colors[0])
    plt.xlabel('Average Temperature (°C)', fontsize=11)
    plt.ylabel('Daily Power Usage (GWh)', fontsize=11)
    plt.title('Temperature vs Power Usage', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_temp_vs_power.png', dpi=150)
    plt.close()
    
    # 2. 월별 추이
    print("  [2/8] 월별 추이...")
    monthly_avg = df.groupby(df['일자'].dt.to_period('M')).agg({
        '일별_전력사용량_KWH': 'mean',
        '평균기온(°C)': 'mean'
    }).reset_index()
    monthly_avg['일자'] = monthly_avg['일자'].astype(str)
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(range(len(monthly_avg)), monthly_avg['일별_전력사용량_KWH']/1000000, 
            alpha=0.7, color=colors[0])
    ax1.set_ylabel('Power Usage (GWh)', fontsize=11)
    ax1.set_xticks(range(len(monthly_avg)))
    ax1.set_xticklabels(monthly_avg['일자'], rotation=45, ha='right')
    
    ax2 = ax1.twinx()
    ax2.plot(range(len(monthly_avg)), monthly_avg['평균기온(°C)'], 
             color=colors[1], marker='o', linewidth=2, markersize=6)
    ax2.set_ylabel('Temperature (°C)', fontsize=11, color=colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])
    
    plt.title('Monthly Power Usage and Temperature Trend', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_monthly_trend.png', dpi=150)
    plt.close()
    
    # 3. 계절별 박스플롯
    print("  [3/8] 계절별 박스플롯...")
    df['월'] = df['일자'].dt.month
    def get_season(month):
        if month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        elif month in [9, 10, 11]: return 'Fall'
        else: return 'Winter'
    df['Season'] = df['월'].apply(get_season)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(data=df, x='Season', y='일별_전력사용량_KWH',
                order=['Spring', 'Summer', 'Fall', 'Winter'], 
                palette='Set2', ax=axes[0])
    axes[0].set_ylabel('Daily Power Usage (kWh)', fontsize=11)
    axes[0].set_title('Power Usage by Season', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 5000000)
    
    sns.boxplot(data=df, x='Season', y='평균기온(°C)',
                order=['Spring', 'Summer', 'Fall', 'Winter'],
                palette='Set3', ax=axes[1])
    axes[1].set_ylabel('Temperature (°C)', fontsize=11)
    axes[1].set_title('Temperature by Season', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_seasonal_boxplot.png', dpi=150)
    plt.close()
    
    # 4. 상관관계 히트맵
    print("  [4/8] 상관관계 히트맵...")
    corr_cols = ['일별_전력사용량_KWH', '평균기온(°C)', '최저기온(°C)', 
                 '최고기온(°C)', '일강수량(mm)', '평균 풍속(m/s)', '평균 상대습도(%)']
    corr_matrix = df[corr_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Correlation Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_correlation_heatmap.png', dpi=150)
    plt.close()
    
    # 5. 요일별 평균
    print("  [5/8] 요일별 평균...")
    df['요일'] = df['일자'].dt.dayofweek
    weekday_avg = df.groupby('요일')['일별_전력사용량_KWH'].mean()
    
    plt.figure(figsize=(10, 6))
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    bars = plt.bar(weekday_names, weekday_avg/1000000, color=colors[:7])
    bars[5].set_color(colors[8])  # 토요일
    bars[6].set_color(colors[8])  # 일요일
    
    plt.ylabel('Average Power Usage (GWh)', fontsize=11)
    plt.title('Power Usage by Day of Week', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/5_weekday_usage.png', dpi=150)
    plt.close()
    
    # 6. 기온대별 전력 사용량
    print("  [6/8] 기온대별 전력 사용량...")
    temp_bins = [-20, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40]
    temp_labels = ['<-10', '-10~-5', '-5~0', '0~5', '5~10', 
                   '10~15', '15~20', '20~25', '25~30', '>30']
    df['Temp Range'] = pd.cut(df['평균기온(°C)'], bins=temp_bins, labels=temp_labels)
    
    temp_group = df.groupby('Temp Range')['일별_전력사용량_KWH'].agg(['mean', 'count']).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(range(len(temp_group)), temp_group['mean']/1000000, 
            alpha=0.7, color=colors[0])
    ax1.set_xlabel('Temperature Range (°C)', fontsize=11)
    ax1.set_ylabel('Average Power Usage (GWh)', fontsize=11)
    ax1.set_xticks(range(len(temp_group)))
    ax1.set_xticklabels(temp_group['Temp Range'], rotation=45)
    
    ax2 = ax1.twinx()
    ax2.plot(range(len(temp_group)), temp_group['count'], 
             color=colors[1], marker='o', linewidth=2, markersize=8)
    ax2.set_ylabel('Number of Days', fontsize=11, color=colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])
    
    plt.title('Power Usage by Temperature Range', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/6_temp_range_usage.png', dpi=150)
    plt.close()
    
    # 7. 시계열 (최근 6개월)
    print("  [7/8] 시계열 그래프...")
    recent_data = df[df['일자'] >= df['일자'].max() - pd.Timedelta(days=180)].copy()
    
    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.plot(recent_data['일자'], recent_data['일별_전력사용량_KWH']/1000000, 
             color=colors[0], linewidth=1, alpha=0.7)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Power Usage (GWh)', fontsize=11)
    
    ax2 = ax1.twinx()
    ax2.plot(recent_data['일자'], recent_data['평균기온(°C)'], 
             color=colors[1], linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('Temperature (°C)', fontsize=11, color=colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])
    
    plt.title('Time Series (Recent 6 months)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/7_timeseries.png', dpi=150)
    plt.close()
    
    # 8. 지역별 Top 10
    print("  [8/8] 지역별 전력 사용량...")
    dong_avg = df.groupby('법정동명').agg({
        '일별_전력사용량_KWH': ['mean', 'count']
    }).reset_index()
    dong_avg.columns = ['법정동명', 'avg_power', 'count']
    dong_avg = dong_avg[dong_avg['count'] >= 100]
    dong_avg = dong_avg.sort_values('avg_power', ascending=False).head(10)
    dong_avg['법정동명_short'] = dong_avg['법정동명'].str.replace('서울특별시 종로구 ', '')
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(dong_avg)), dong_avg['avg_power']/1000000, color=colors[:10])
    plt.yticks(range(len(dong_avg)), dong_avg['법정동명_short'])
    plt.xlabel('Average Power Usage (GWh)', fontsize=11)
    plt.title('Top 10 Districts by Power Usage', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2.,
                 f'{width:.2f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/8_district_top10.png', dpi=150)
    plt.close()
    
    print(f"\n✓ 시각화 완료! '{output_dir}' 폴더에 8개 파일 생성됨")


# ============================================================================
# 4. 맞춤 분석
# ============================================================================

def custom_analysis(data_file):
    """맞춤 분석"""
    print("\n" + "="*100)
    print("🔍 맞춤 분석")
    print("="*100)
    
    df = pd.read_csv(data_file)
    df['일자'] = pd.to_datetime(df['일자'])
    
    print("\n분석 옵션:")
    print("1. 특정 기간 분석")
    print("2. 특정 지역 분석")
    print("3. 기온 범위별 분석")
    print("4. 상위/하위 전력 사용일 분석")
    
    choice = input("\n선택 (1-4): ").strip()
    
    if choice == '1':
        print("\n기간 입력 예: 2023-07-01")
        start = input("시작일: ").strip()
        end = input("종료일: ").strip()
        
        mask = (df['일자'] >= start) & (df['일자'] <= end)
        period_data = df[mask]
        
        print(f"\n{start} ~ {end} 분석 결과:")
        print(f"  데이터 개수: {len(period_data):,}개")
        print(f"  평균 전력: {period_data['일별_전력사용량_KWH'].mean():,.0f} kWh")
        print(f"  최대 전력: {period_data['일별_전력사용량_KWH'].max():,.0f} kWh")
        print(f"  최소 전력: {period_data['일별_전력사용량_KWH'].min():,.0f} kWh")
        print(f"  평균 기온: {period_data['평균기온(°C)'].mean():.1f}°C")
        print(f"  총 강수량: {period_data['일강수량(mm)'].sum():.1f}mm")
        
    elif choice == '2':
        print("\n사용 가능한 지역 (샘플):")
        regions = df['법정동명'].unique()[:10]
        for i, region in enumerate(regions, 1):
            print(f"  {i}. {region}")
        
        region_name = input("\n지역명 입력 (일부만 입력 가능): ").strip()
        region_data = df[df['법정동명'].str.contains(region_name)]
        
        print(f"\n'{region_name}' 포함 지역 분석 결과:")
        print(f"  데이터 개수: {len(region_data):,}개")
        print(f"  평균 전력: {region_data['일별_전력사용량_KWH'].mean():,.0f} kWh")
        print(f"  최대 전력: {region_data['일별_전력사용량_KWH'].max():,.0f} kWh")
        print(f"  최소 전력: {region_data['일별_전력사용량_KWH'].min():,.0f} kWh")
        
    elif choice == '3':
        min_temp = float(input("최저 기온: "))
        max_temp = float(input("최고 기온: "))
        
        temp_data = df[(df['평균기온(°C)'] >= min_temp) & (df['평균기온(°C)'] <= max_temp)]
        
        print(f"\n기온 {min_temp}~{max_temp}°C 분석 결과:")
        print(f"  해당 일수: {len(temp_data):,}일")
        print(f"  평균 전력: {temp_data['일별_전력사용량_KWH'].mean():,.0f} kWh")
        print(f"  평균 기온: {temp_data['평균기온(°C)'].mean():.1f}°C")
        print(f"  평균 강수량: {temp_data['일강수량(mm)'].mean():.1f}mm")
        
    elif choice == '4':
        n = int(input("몇 개 출력? (예: 10): "))
        
        print(f"\n전력 사용량 상위 {n}개:")
        top_n = df.nlargest(n, '일별_전력사용량_KWH')[
            ['일자', '법정동명', '일별_전력사용량_KWH', '평균기온(°C)', '일강수량(mm)']
        ]
        print(top_n.to_string(index=False))
        
        print(f"\n전력 사용량 하위 {n}개:")
        bottom_n = df.nsmallest(n, '일별_전력사용량_KWH')[
            ['일자', '법정동명', '일별_전력사용량_KWH', '평균기온(°C)', '일강수량(mm)']
        ]
        print(bottom_n.to_string(index=False))


# ============================================================================
# 5. 전체 실행
# ============================================================================

def run_all(data_file):
    """전체 실행"""
    print("\n⚡ 전체 분석 및 시각화를 실행합니다...\n")
    
    basic_analysis(data_file)
    input("\n계속하려면 Enter를 누르세요...")
    
    detailed_analysis(data_file)
    input("\n계속하려면 Enter를 누르세요...")
    
    visualization(data_file)
    
    print("\n✓ 모든 작업 완료!")


# ============================================================================
# 프로그램 실행
# ============================================================================

if __name__ == "__main__":
    main()
