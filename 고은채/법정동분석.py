"""
법정동 데이터 분석 프로그램
==========================

데이터: 법정동_전처리_완료.csv
지역: 서울특별시 전체 법정동 (1,112개)

필요한 패키지:
pip install pandas numpy matplotlib seaborn

사용법:
python 법정동분석.py
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
    print("법정동 데이터 분석 프로그램")
    print("="*100)
    
    data_file = '법정동_전처리_완료.csv'
    
    if not os.path.exists(data_file):
        print(f"\n❌ '{data_file}' 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n✓ 데이터 파일 발견: {data_file}")
    
    # 메뉴
    while True:
        print("\n" + "="*100)
        print("작업 선택:")
        print("="*100)
        print("\n1. 📊 기본 정보")
        print("2. 📈 구별 통계")
        print("3. 🔍 법정동 검색")
        print("4. 📋 폐지 법정동 분석")
        print("5. 💾 결과 저장")
        print("0. ❌ 종료")
        
        choice = input("\n선택: ").strip()
        
        if choice == '1':
            basic_info(data_file)
        elif choice == '2':
            district_stats(data_file)
        elif choice == '3':
            search_dong(data_file)
        elif choice == '4':
            analyze_closed(data_file)
        elif choice == '5':
            save_results(data_file)
        elif choice == '0':
            print("\n종료합니다.")
            break
        else:
            print("\n❌ 잘못된 선택입니다.")


# ============================================================================
# 1. 기본 정보
# ============================================================================

def basic_info(data_file):
    """기본 정보"""
    print("\n" + "="*100)
    print("📊 기본 정보")
    print("="*100)
    
    df = pd.read_csv(data_file)
    
    print(f"\n데이터 크기: {len(df):,}행 × {len(df.columns)}열")
    
    print("\n컬럼 정보:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col} ({df[col].dtype})")
    
    print("\n지역 정보:")
    print(f"  총 법정동 수: {len(df):,}개")
    print(f"  시군구 수: {df['시군구명'].nunique()}개")
    print(f"  현존 법정동: {len(df[df['폐지구분'] == '현존']):,}개")
    print(f"  폐지 법정동: {len(df[df['폐지구분'] != '현존']):,}개")
    
    print("\n시군구 목록:")
    districts = df['시군구명'].value_counts().sort_index()
    for district, count in districts.items():
        if district != '서울특별시':  # 최상위 제외
            print(f"  {district}: {count}개")
    
    print("\n샘플 데이터:")
    print(df.head(10).to_string(index=False))
    
    print("\n✓ 완료!")


# ============================================================================
# 2. 구별 통계
# ============================================================================

def district_stats(data_file):
    """구별 통계"""
    print("\n" + "="*100)
    print("📈 구별 통계")
    print("="*100)
    
    df = pd.read_csv(data_file)
    
    # 최상위 제외
    df_detail = df[df['법정동_세부코드'] != 0].copy()
    
    # 구별 집계
    print("\n[1] 구별 법정동 수")
    district_count = df_detail.groupby('시군구명').size().sort_values(ascending=False)
    
    for i, (district, count) in enumerate(district_count.items(), 1):
        print(f"  {i:2d}. {district:20s}: {count:3d}개")
    
    # 구별 폐지 현황
    print("\n[2] 구별 폐지 법정동 수")
    closed = df_detail[df_detail['폐지구분'] != '현존']
    closed_count = closed.groupby('시군구명').size().sort_values(ascending=False)
    
    for i, (district, count) in enumerate(closed_count.items(), 1):
        total = district_count[district]
        pct = (count / total) * 100
        print(f"  {i:2d}. {district:20s}: {count:3d}개 ({pct:5.1f}%)")
    
    # 구별 현존 법정동
    print("\n[3] 구별 현존 법정동 수")
    active = df_detail[df_detail['폐지구분'] == '현존']
    active_count = active.groupby('시군구명').size().sort_values(ascending=False)
    
    for i, (district, count) in enumerate(active_count.items(), 1):
        print(f"  {i:2d}. {district:20s}: {count:3d}개")
    
    print("\n✓ 완료!")


# ============================================================================
# 3. 법정동 검색
# ============================================================================

def search_dong(data_file):
    """법정동 검색"""
    print("\n" + "="*100)
    print("🔍 법정동 검색")
    print("="*100)
    
    df = pd.read_csv(data_file)
    
    print("\n검색 방법:")
    print("1. 법정동명으로 검색")
    print("2. 시군구로 검색")
    print("3. 법정동코드로 검색")
    
    choice = input("\n선택: ").strip()
    
    if choice == '1':
        keyword = input("\n법정동명 입력 (예: 청운동): ").strip()
        results = df[df['법정동명'].str.contains(keyword, na=False)]
        
        print(f"\n'{keyword}' 검색 결과: {len(results)}개")
        if len(results) > 0:
            print(results[['법정동코드', '법정동명', '폐지구분', '시군구명']].to_string(index=False))
        else:
            print("검색 결과가 없습니다.")
    
    elif choice == '2':
        district = input("\n시군구명 입력 (예: 종로구): ").strip()
        results = df[df['시군구명'].str.contains(district, na=False)]
        
        print(f"\n'{district}' 검색 결과: {len(results)}개")
        if len(results) > 0:
            # 현존만
            active = results[results['폐지구분'] == '현존']
            print(f"\n현존 법정동 ({len(active)}개):")
            print(active[['법정동명_세부', '법정동코드', '법정동_세부코드']].to_string(index=False))
    
    elif choice == '3':
        code = input("\n법정동코드 입력 (10자리): ").strip()
        results = df[df['법정동코드'].astype(str) == code]
        
        print(f"\n검색 결과: {len(results)}개")
        if len(results) > 0:
            print(results.to_string(index=False))
        else:
            print("검색 결과가 없습니다.")


# ============================================================================
# 4. 폐지 법정동 분석
# ============================================================================

def analyze_closed(data_file):
    """폐지 법정동 분석"""
    print("\n" + "="*100)
    print("📋 폐지 법정동 분석")
    print("="*100)
    
    df = pd.read_csv(data_file)
    
    # 폐지된 법정동
    closed = df[df['폐지구분'] != '현존']
    
    print(f"\n총 폐지 법정동: {len(closed):,}개")
    
    print("\n폐지 구분:")
    closed_types = closed['폐지구분'].value_counts()
    for type_name, count in closed_types.items():
        print(f"  {type_name}: {count}개")
    
    print("\n구별 폐지 법정동 수:")
    district_closed = closed.groupby('시군구명').size().sort_values(ascending=False)
    for i, (district, count) in enumerate(district_closed.items(), 1):
        if district != '서울특별시':
            print(f"  {i:2d}. {district:20s}: {count:3d}개")
    
    print("\n폐지 법정동 목록 (상위 20개):")
    print(closed[['법정동명', '폐지구분', '시군구명']].head(20).to_string(index=False))
    
    print("\n✓ 완료!")


# ============================================================================
# 5. 결과 저장
# ============================================================================

def save_results(data_file):
    """결과 저장"""
    print("\n" + "="*100)
    print("💾 결과 저장")
    print("="*100)
    
    df = pd.read_csv(data_file)
    
    # 1. 구별 통계
    df_detail = df[df['법정동_세부코드'] != 0]
    district_stats = df_detail.groupby('시군구명').agg({
        '법정동코드': 'count',
        '폐지구분': lambda x: (x == '현존').sum()
    }).rename(columns={'법정동코드': '총_법정동수', '폐지구분': '현존_법정동수'})
    district_stats['폐지_법정동수'] = district_stats['총_법정동수'] - district_stats['현존_법정동수']
    district_stats.to_csv('구별_통계.csv', encoding='utf-8-sig')
    print("  ✓ 구별_통계.csv")
    
    # 2. 현존 법정동 목록
    active = df[df['폐지구분'] == '현존']
    active.to_csv('현존_법정동목록.csv', index=False, encoding='utf-8-sig')
    print("  ✓ 현존_법정동목록.csv")
    
    # 3. 폐지 법정동 목록
    closed = df[df['폐지구분'] != '현존']
    closed.to_csv('폐지_법정동목록.csv', index=False, encoding='utf-8-sig')
    print("  ✓ 폐지_법정동목록.csv")
    
    # 4. 구별 법정동 목록
    for district in df_detail['시군구명'].unique():
        if district != '서울특별시':
            district_data = df_detail[df_detail['시군구명'] == district]
            filename = f'{district}_법정동목록.csv'
            district_data.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"  ✓ {filename}")
    
    # 5. 시각화
    print("\n그래프 생성 중...")
    
    # 구별 법정동 수
    plt.figure(figsize=(12, 8))
    district_count = df_detail.groupby('시군구명').size().sort_values(ascending=False)
    
    colors = sns.color_palette('husl', len(district_count))
    bars = plt.barh(range(len(district_count)), district_count.values, color=colors)
    plt.yticks(range(len(district_count)), district_count.index)
    plt.xlabel('Number of Dong', fontsize=11)
    plt.title('Number of Beobjeongdong by District', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # 값 표시
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                 f'{int(width)}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('구별_법정동수.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 구별_법정동수.png")
    
    # 현존/폐지 비율
    active_count = df[df['폐지구분'] == '현존'].groupby('시군구명').size()
    total_count = df.groupby('시군구명').size()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = range(len(active_count))
    
    ax.barh(x, total_count[active_count.index], alpha=0.5, label='Closed', color='gray')
    ax.barh(x, active_count.values, label='Active', color='green')
    
    ax.set_yticks(x)
    ax.set_yticklabels(active_count.index)
    ax.set_xlabel('Number of Dong', fontsize=11)
    ax.set_title('Active vs Closed Beobjeongdong', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('현존_폐지_비율.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 현존_폐지_비율.png")
    
    print("\n✓ 모든 결과 저장 완료!")


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    main()
