import pandas as pd

# T1.0 폴더 안에 있는 data.csv를 읽어오도록 경로 수정 (슬래시 주의)
df = pd.read_csv('T1.0/data.csv')

# 데이터 구조를 파악하기 위한 출력 코드 추가
print("=== 컬럼(열) 이름 확인 ===")
print(df.columns)

print("\n=== 데이터 위에서 5줄 확인 ===")
print(df.head())
print(df.info())

df_2 = pd.read_csv('T1.0/target_position.csv')
print("=== 컬럼(열) 이름 확인 ===")
print(df_2.columns)

print("\n=== 데이터 위에서 5줄 확인 ===")
print(df_2.head())
print(df_2.info())