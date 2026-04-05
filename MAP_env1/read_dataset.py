import pandas as pd
import numpy as np
import math
from pathlib import Path

def load_all_avalanche_data(base_dir_path, noise_std=5.0):
    """
    모든 T*.* 폴더를 순회하며 data.csv와 target_position.csv를 읽어
    하나의 통합된 데이터셋(X, Y)으로 만드는 함수
    """
    base_dir = Path(base_dir_path)
    all_processed_data = []
    
    print(f"📁 베이스 디렉토리 탐색 중: {base_dir}")
    
    # 베이스 디렉토리 안의 모든 폴더(T1.0, T2.1 등)를 순회
    for folder_path in base_dir.iterdir():
        if not folder_path.is_dir() or not folder_path.name.startswith('T'):
            continue
            
        print(f"  👉 {folder_path.name} 폴더 처리 중...")
        
        data_csv_path = folder_path / 'data.csv'
        target_csv_path = folder_path / 'target_position.csv'
        
        # 필수 파일이 없으면 스킵
        if not data_csv_path.exists() or not target_csv_path.exists():
            print(f"    ⚠️ 경고: {folder_path.name} 폴더에 필수 CSV 파일이 없습니다. 스킵합니다.")
            continue
            
        # 1. 조난자(Target) 위치 읽기 (오타 수정 완료: pd.read_csv)
        target_df = pd.read_csv(target_csv_path)
        
        # target_position.csv의 컬럼명에 맞게 위도/경도 추출
        try:
            target_lat = target_df['latitude'].values[0] 
            target_lon = target_df['longitude'].values[0]
        except KeyError:
            # 혹시 target_position.csv의 컬럼명이 다를 경우를 대비한 방어 코드
            cols = target_df.columns
            target_lat = target_df[cols[0]].values[0]
            target_lon = target_df[cols[1]].values[0]
            
        target_alt = 1870.0 # 돌로미티 산맥 고정 고도 (필요시 수정)

        # 2. 드론 데이터(RSSI, SNR, 위치) 읽기
        df = pd.read_csv(data_csv_path)
        
        # 3. 폴더 식별자 추가 (나중에 어떤 테스트 데이터인지 구분하기 위해)
        df['run_id'] = folder_path.name
        
        # 4. AoA 계산 및 노이즈 합성 로직
        synthetic_aoa_azimuth = []
        synthetic_aoa_elevation = []
        
        for i in range(len(df)):
            # 💡 앞서 확인한 실제 데이터 컬럼명 적용 ('longitude', 'latitude', 'height[m]')
            dx = (target_lon - df['longitude'].values[i]) * 111000 * math.cos(math.radians(target_lat))
            dy = (target_lat - df['latitude'].values[i]) * 111000
            dz = target_alt - df['height[m]'].values[i] 
            
            true_azimuth = math.degrees(math.atan2(dy, dx))
            horizontal_dist = math.hypot(dx, dy)
            true_elevation = math.degrees(math.atan2(dz, horizontal_dist))
            
            # 가우시안 노이즈(오차) 추가
            synthetic_aoa_azimuth.append(true_azimuth + np.random.normal(0, noise_std))
            synthetic_aoa_elevation.append(true_elevation + np.random.normal(0, noise_std))
            
        df['noisy_azimuth'] = synthetic_aoa_azimuth
        df['noisy_elevation'] = synthetic_aoa_elevation
        
        # target_lat, lon 값을 df에도 복사해두면 나중에 정답(Y) 라벨 만들 때 편함
        df['target_lat'] = target_lat
        df['target_lon'] = target_lon
        
        all_processed_data.append(df)

    # 5. 모든 폴더의 데이터를 하나의 거대한 DataFrame으로 병합
    if not all_processed_data:
        print("❌ 처리할 데이터가 없습니다. 폴더 경로를 확인하세요.")
        return None, None, None
        
    final_df = pd.concat(all_processed_data, ignore_index=True)
    print(f"✅ 총 {len(final_df)} 줄의 데이터 병합 및 전처리 완료!")
    
    # 6. ANN 입력(X) 및 정답(Y) 추출 (실제 컬럼명 반영: 'rssi[dBm]', 'snr[dB]')
    X = final_df[['rssi[dBm]', 'snr[dB]', 'noisy_azimuth', 'noisy_elevation']].values
    Y = final_df[['target_lon', 'target_lat']].values 
    
    return X, Y, final_df


if __name__ == "__main__":
    # 💡 [필수 수정] 다운로드 받은 전체 데이터셋 폴더(T1.0, T2.1 등이 들어있는 최상위 폴더)의 절대 경로를 입력하세요.
    # 윈도우 경로를 붙여넣으실 때는 경로 맨 앞에 r 을 붙여주시면 백슬래시(\) 오류가 나지 않습니다.
    BASE_DIR = Path(__file__).resolve().parent
    base_directory = BASE_DIR / "dataset"
    
    X_data, Y_data, full_dataframe = load_all_avalanche_data(base_directory)
    
    if full_dataframe is not None:
        print("\n=== 최종 병합된 데이터 상위 5줄 (핵심 데이터만) ===")
        # 보기 편하게 중요한 컬럼만 뽑아서 출력
        print(full_dataframe[['run_id', 'rssi[dBm]', 'snr[dB]', 'noisy_azimuth', 'noisy_elevation', 'target_lat']].head())
        print(f"\n=== 전체 데이터 크기 (행, 열) ===\n{full_dataframe.shape}")