import pandas as pd

# 1. 두 개의 파일 각각 불러오기
df = pd.read_csv('T1.0/data.csv')
df_2 = pd.read_csv('T1.0/target_position.csv') #정답지

# 2. df_2(정답지)에서 조난자의 위도, 경도 값을 뽑아옵니다. (.iloc[0]은 첫 번째 줄의 값을 가져온다는 뜻입니다)
target_lat = df_2['latitude'].iloc[0]
target_lon = df_2['longitude'].iloc[0]

# 3. 수천 줄짜리 드론 데이터(df)의 새로운 열(Column)로 조난자 위치를 통째로 채워 넣습니다.
df['target_lat'] = target_lat #위도
df['target_lon'] = target_lon # 경도

# 4. 결과 확인
print("\n=== 정답지가 결합된 최종 데이터 위에서 5줄 ===")
print(df[['rssi[dBm]', 'latitude', 'longitude', 'target_lat', 'target_lon']].head())