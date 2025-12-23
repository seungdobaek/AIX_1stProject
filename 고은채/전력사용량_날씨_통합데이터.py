"""
전력 사용량 & 날씨 데이터 통합 분석 프로그램
============================================

데이터: 최종_통합_데이터_완벽.csv
기간: 2022-06-28 ~ 2024-01-28
지역: 서울특별시 종로구

필요한 패키지:
pip install pandas numpy matplotlib seaborn

사용법:
python 전력분석.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    print("전력 사용량 & 날씨 데이터 분석 프로그램")
    print("="*100)
    
    # 데이터 파일
    data_file = '최종_통합_데이터_완벽.csv'
    
    if not os.path.exists(data_file):
        print(f"\n❌ '{data_file}' 파일을 찾을 수 없습니다.")
        print("현재 폴더에 데이터 파일을 넣어주세요.")
        return
    
    print(f"\n✓ 데이터 파일 발견: {data_file}")
    
    # 메뉴
    while True:
        print("\n" + "="*100)
        print("작업 선택:")
        print("="*100)
        print("\n1. 📊 기본 통계")
        print("2. 📈 상세 분석 (월별/계절별/요일별/지역별)")
        print("3. 🎨 시각화 (그래프 10개)")
        print("4. 🔍 맞춤 분석")
        print("5. ⚡ 전체 실행")
        print("0. ❌ 종료")
        
        choice = input("\n선택: ").strip()
        
        if choice == '1':
            basic_stats(data_file)
        elif choice == '2':
            detailed_analysis(data_file)
        elif choice == '3':
            create_visualizations(data_file)
        elif choice == '4':
            custom_analysis(data_file)
        elif choice == '5':
            run_all(data_file)
        elif choice == '0':
            print("\n종료합니다.")
            break
        else:
            print("\n❌ 잘못된 선택입니다.")


# ============================================================================
# 1. 기본 통계
# ============================================================================

def basic_stats(data_file):
    """기본 통계"""
    print("\n" + "="*100)
    print("📊 기본 통계")
    print("="*100)
    
    df = pd.read_csv(data_file)
    df['일자'] = pd.to_datetime(df['일자'])
    
    print(f"\n데이터 크기: {len(df):,}행 × {len(df.columns)}열")
    print(f"기간: {df['일자'].min().date()} ~ {df['일자'].max().date()}")
    print(f"결측치: {df.isnull().sum().sum()}개")
    
    print(f"\n지역 정보:")
    print(f"  시군구: {df['시군구명'].unique()[0]}")
    print(f"  법정동 개수: {df['법정동명'].nunique()}개")
    
    print("\n주요 변수 통계:")
    stats_cols = ['일별_전력사용량_KWH', '평균기온(°C)', '최저기온(°C)', 
                  '최고기온(°C)', '일강수량(mm)', '평균 풍속(m/s)', '평균 상대습도(%)']
    print(df[stats_cols].describe().round(2))
    
    print("\n상관관계:")
    corr = df[stats_cols].corr()['일별_전력사용량_KWH'].sort_values(ascending=False)
    print(corr.round(3))
    
    print("\n✓ 완료!")


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
    
    # 계절
    def get_season(month):
        if month in [3, 4, 5]: return '봄'
        elif month in [6, 7, 8]: return '여름'
        elif month in [9, 10, 11]: return '가을'
        else: return '겨울'
    df['계절'] = df['월'].apply(get_season)
    
    # 1. 계절별
    print("\n[1] 계절별 통계")
    seasonal = df.groupby('계절').agg({
        '일별_전력사용량_KWH': ['mean', 'std', 'count'],
        '평균기온(°C)': 'mean',
        '일강수량(mm)': 'sum'
    }).round(0)
    print(seasonal)
    
    # 2. 요일별
    print("\n[2] 요일별 통계")
    weekdays = ['월', '화', '수', '목', '금', '토', '일']
    weekday_stats = df.groupby('요일')['일별_전력사용량_KWH'].agg(['mean', 'count'])
    for i, (mean, count) in enumerate(weekday_stats.values):
        print(f"  {weekdays[i]}: {mean:,.0f} kWh (n={count})")
    
    # 3. 월별
    print("\n[3] 월별 통계 (2023년)")
    monthly = df[df['연도'] == 2023].groupby('월').agg({
        '일별_전력사용량_KWH': 'mean',
        '평균기온(°C)': 'mean'
    }).round(0)
    print(monthly)
    
    # 4. 지역별 Top 10
    print("\n[4] 전력 사용량 Top 10 지역")
    region_avg = df.groupby('법정동명')['일별_전력사용량_KWH'].agg(['mean', 'count'])
    region_avg = region_avg[region_avg['count'] >= 100].sort_values('mean', ascending=False)
    
    for i, (region, (mean, count)) in enumerate(region_avg.head(10).iterrows(), 1):
        region_short = region.replace('서울특별시 종로구 ', '')
        print(f"  {i:2d}. {region_short:15s}: {mean:10,.0f} kWh")
    
    # 5. 극한 기상
    print("\n[5] 극한 기상 조건")
    hot = df[df['평균기온(°C)'] > 28]
    cold = df[df['평균기온(°C)'] < 0]
    rain = df[df['일강수량(mm)'] > 50]
    normal = df[(df['평균기온(°C)'] >= 10) & (df['평균기온(°C)'] <= 25) & (df['일강수량(mm)'] < 10)]
    
    print(f"  폭염 (>28°C): {hot['일별_전력사용량_KWH'].mean():,.0f} kWh (n={len(hot)})")
    print(f"  한파 (<0°C): {cold['일별_전력사용량_KWH'].mean():,.0f} kWh (n={len(cold)})")
    print(f"  폭우 (>50mm): {rain['일별_전력사용량_KWH'].mean():,.0f} kWh (n={len(rain)})")
    print(f"  일반: {normal['일별_전력사용량_KWH'].mean():,.0f} kWh (n={len(normal)})")
    
    # 6. 기온대별
    print("\n[6] 기온대별 평균")
    df['기온대'] = pd.cut(df['평균기온(°C)'], 
                        bins=[-20, -5, 0, 5, 10, 15, 20, 25, 30, 40],
                        labels=['<-5', '-5~0', '0~5', '5~10', '10~15', '15~20', '20~25', '25~30', '>30'])
    temp_stats = df.groupby('기온대')['일별_전력사용량_KWH'].agg(['mean', 'count'])
    print(temp_stats.round(0))
    
    # 저장
    print("\n[7] 결과 저장")
    seasonal.to_csv('분석_계절별.csv', encoding='utf-8-sig')
    monthly.to_csv('분석_월별.csv', encoding='utf-8-sig')
    region_avg.to_csv('분석_지역별.csv', encoding='utf-8-sig')
    print("  ✓ 분석_계절별.csv")
    print("  ✓ 분석_월별.csv")
    print("  ✓ 분석_지역별.csv")
    
    print("\n✓ 완료!")


# ============================================================================
# 3. 시각화
# ============================================================================

def create_visualizations(data_file):
    """시각화 생성"""
    print("\n" + "="*100)
    print("🎨 시각화 생성")
    print("="*100)
    
    df = pd.read_csv(data_file)
    df['일자'] = pd.to_datetime(df['일자'])
    
    output_dir = 'graphs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n그래프 생성 중...")
    colors = sns.color_palette('husl', 10)
    
    # 1. 기온 vs 전력
    print("  [1/10] 기온-전력 산점도")
    plt.figure(figsize=(10, 6))
    plt.scatter(df['평균기온(°C)'], df['일별_전력사용량_KWH']/1000000, 
                alpha=0.3, s=10, c=colors[0])
    plt.xlabel('Temperature (°C)', fontsize=11)
    plt.ylabel('Power Usage (GWh)', fontsize=11)
    plt.title('Temperature vs Power Usage', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_temp_power.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 월별 추이
    print("  [2/10] 월별 추이")
    monthly = df.groupby(df['일자'].dt.to_period('M')).agg({
        '일별_전력사용량_KWH': 'mean',
        '평균기온(°C)': 'mean'
    }).reset_index()
    monthly['일자'] = monthly['일자'].astype(str)
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(range(len(monthly)), monthly['일별_전력사용량_KWH']/1000000, alpha=0.7, color=colors[0])
    ax1.set_ylabel('Power (GWh)', fontsize=11)
    ax1.set_xticks(range(len(monthly)))
    ax1.set_xticklabels(monthly['일자'], rotation=45, ha='right')
    
    ax2 = ax1.twinx()
    ax2.plot(range(len(monthly)), monthly['평균기온(°C)'], 
             color=colors[1], marker='o', linewidth=2)
    ax2.set_ylabel('Temp (°C)', fontsize=11, color=colors[1])
    
    plt.title('Monthly Trend', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_monthly.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 계절별 박스플롯
    print("  [3/10] 계절별 박스플롯")
    df['월'] = df['일자'].dt.month
    def get_season(m):
        if m in [3,4,5]: return 'Spring'
        elif m in [6,7,8]: return 'Summer'
        elif m in [9,10,11]: return 'Fall'
        else: return 'Winter'
    df['Season'] = df['월'].apply(get_season)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Season', y='일별_전력사용량_KWH',
                order=['Spring', 'Summer', 'Fall', 'Winter'], palette='Set2')
    plt.ylabel('Power Usage (kWh)', fontsize=11)
    plt.title('Seasonal Power Usage', fontsize=13, fontweight='bold')
    plt.ylim(0, 5000000)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_seasonal.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 요일별
    print("  [4/10] 요일별 평균")
    df['요일'] = df['일자'].dt.dayofweek
    weekday_avg = df.groupby('요일')['일별_전력사용량_KWH'].mean()
    
    plt.figure(figsize=(10, 6))
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    bars = plt.bar(weekdays, weekday_avg/1000000, color=colors[:7])
    bars[5].set_color(colors[8])
    bars[6].set_color(colors[8])
    
    plt.ylabel('Power (GWh)', fontsize=11)
    plt.title('Weekday Power Usage', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h, f'{h:.2f}',
                 ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_weekday.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. 지역별 Top 10
    print("  [5/10] 지역별 Top 10")
    region = df.groupby('법정동명').agg({
        '일별_전력사용량_KWH': ['mean', 'count']
    }).reset_index()
    region.columns = ['법정동명', 'mean', 'count']
    region = region[region['count'] >= 100].sort_values('mean', ascending=False).head(10)
    region['short'] = region['법정동명'].str.replace('서울특별시 종로구 ', '')
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(region)), region['mean']/1000000, color=colors[:10])
    plt.yticks(range(len(region)), region['short'])
    plt.xlabel('Power (GWh)', fontsize=11)
    plt.title('Top 10 Districts', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/5_districts.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. 상관관계 히트맵
    print("  [6/10] 상관관계")
    corr_cols = ['일별_전력사용량_KWH', '평균기온(°C)', '최저기온(°C)', 
                 '최고기온(°C)', '일강수량(mm)', '평균 풍속(m/s)', '평균 상대습도(%)']
    corr = df[corr_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/6_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 7. 기온 분포
    print("  [7/10] 기온 분포")
    plt.figure(figsize=(10, 6))
    plt.hist(df['평균기온(°C)'], bins=50, alpha=0.7, edgecolor='black', color=colors[0])
    plt.xlabel('Temperature (°C)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Temperature Distribution', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/7_temp_dist.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 8. 전력 분포
    print("  [8/10] 전력 분포")
    plt.figure(figsize=(10, 6))
    plt.hist(df['일별_전력사용량_KWH']/1000000, bins=50, alpha=0.7, 
             edgecolor='black', color=colors[1])
    plt.xlabel('Power (GWh)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Power Usage Distribution', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/8_power_dist.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 9. 시계열 (최근 3개월)
    print("  [9/10] 시계열")
    recent = df[df['일자'] >= df['일자'].max() - pd.Timedelta(days=90)].copy()
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(recent['일자'], recent['일별_전력사용량_KWH']/1000000, 
             color=colors[0], linewidth=1, alpha=0.7)
    ax1.set_ylabel('Power (GWh)', fontsize=11)
    
    ax2 = ax1.twinx()
    ax2.plot(recent['일자'], recent['평균기온(°C)'], 
             color=colors[1], linewidth=1.5)
    ax2.set_ylabel('Temp (°C)', fontsize=11, color=colors[1])
    
    plt.title('Time Series (Recent 3 months)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/9_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 10. 기온대별
    print("  [10/10] 기온대별")
    df['기온대'] = pd.cut(df['평균기온(°C)'], 
                        bins=[-20, -5, 0, 5, 10, 15, 20, 25, 30, 40],
                        labels=['<-5', '-5~0', '0~5', '5~10', '10~15', '15~20', '20~25', '25~30', '>30'])
    temp_group = df.groupby('기온대')['일별_전력사용량_KWH'].agg(['mean', 'count']).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(range(len(temp_group)), temp_group['mean']/1000000, 
            alpha=0.7, color=colors[0])
    ax1.set_ylabel('Power (GWh)', fontsize=11)
    ax1.set_xticks(range(len(temp_group)))
    ax1.set_xticklabels(temp_group['기온대'], rotation=45)
    
    ax2 = ax1.twinx()
    ax2.plot(range(len(temp_group)), temp_group['count'], 
             color=colors[1], marker='o', linewidth=2, markersize=8)
    ax2.set_ylabel('Count', fontsize=11, color=colors[1])
    
    plt.title('Power by Temperature Range', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/10_temp_range.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 완료! '{output_dir}' 폴더에 10개 파일 생성")


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
    
    print("\n옵션:")
    print("1. 특정 기간")
    print("2. 특정 지역")
    print("3. 기온 범위")
    print("4. Top/Bottom N")
    
    choice = input("\n선택: ").strip()
    
    if choice == '1':
        start = input("시작일 (예: 2023-07-01): ").strip()
        end = input("종료일 (예: 2023-07-31): ").strip()
        
        data = df[(df['일자'] >= start) & (df['일자'] <= end)]
        
        print(f"\n{start} ~ {end}")
        print(f"  데이터: {len(data):,}개")
        print(f"  평균 전력: {data['일별_전력사용량_KWH'].mean():,.0f} kWh")
        print(f"  최대 전력: {data['일별_전력사용량_KWH'].max():,.0f} kWh")
        print(f"  평균 기온: {data['평균기온(°C)'].mean():.1f}°C")
        print(f"  총 강수: {data['일강수량(mm)'].sum():.1f}mm")
        
    elif choice == '2':
        print("\n지역 샘플:")
        for i, r in enumerate(df['법정동명'].unique()[:10], 1):
            print(f"  {i}. {r}")
        
        region = input("\n지역명 (일부): ").strip()
        data = df[df['법정동명'].str.contains(region)]
        
        print(f"\n'{region}' 포함 지역")
        print(f"  데이터: {len(data):,}개")
        print(f"  평균: {data['일별_전력사용량_KWH'].mean():,.0f} kWh")
        print(f"  최대: {data['일별_전력사용량_KWH'].max():,.0f} kWh")
        
    elif choice == '3':
        min_t = float(input("최저 기온: "))
        max_t = float(input("최고 기온: "))
        
        data = df[(df['평균기온(°C)'] >= min_t) & (df['평균기온(°C)'] <= max_t)]
        
        print(f"\n{min_t}~{max_t}°C")
        print(f"  일수: {len(data):,}일")
        print(f"  평균 전력: {data['일별_전력사용량_KWH'].mean():,.0f} kWh")
        print(f"  평균 기온: {data['평균기온(°C)'].mean():.1f}°C")
        
    elif choice == '4':
        n = int(input("개수: "))
        
        print(f"\nTop {n}:")
        top = df.nlargest(n, '일별_전력사용량_KWH')[
            ['일자', '법정동명', '일별_전력사용량_KWH', '평균기온(°C)']
        ]
        print(top.to_string(index=False))
        
        print(f"\nBottom {n}:")
        bottom = df.nsmallest(n, '일별_전력사용량_KWH')[
            ['일자', '법정동명', '일별_전력사용량_KWH', '평균기온(°C)']
        ]
        print(bottom.to_string(index=False))


# ============================================================================
# 5. 전체 실행
# ============================================================================

def run_all(data_file):
    """전체 실행"""
    print("\n⚡ 전체 실행\n")
    
    basic_stats(data_file)
    input("\nEnter...")
    
    detailed_analysis(data_file)
    input("\nEnter...")
    
    create_visualizations(data_file)
    
    print("\n✓ 완료!")


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    main()
