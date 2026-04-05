## 📝 프로젝트 개요
- **목표**: RSSI(수신 신호 강도) 기반의 신호원 탐색 및 위치 추적 알고리즘 구현
- **핵심 기술**:
  - **강화학습 (RL)**: PPO(Proximal Policy Optimization) 알고리즘을 통한 드론 경로 최적화
  - **신호 처리**: RSSI 기반 거리 추정 및 칼만 필터(Kalman Filter)를 통한 노이즈 제거
  - **시각화**: 시뮬레이션 환경 구축 및 실시간 드론 비행 경로 시각화 (Gymnasium 환경 활용)

## 📊 데이터셋 (Dataset)
본 프로젝트에서 사용된 RSSI 및 환경 데이터셋은 대용량 파일 관리 및 접근성을 위해 외부 공인 저장소(Zenodo)를 활용합니다. 

- **Download Link**: [Zenodo - UAV Signal Localization Dataset](https://zenodo.org/records/16572816)
- **파일 구성**: 시뮬레이션 학습에 필요한 신호 강도 로그 및 위치 좌표 데이터

> **Note**: 데이터를 다운로드한 후, 프로젝트 루트 폴더의 `dataset/` 디렉토리에 위치시켜야 `train.py`가 정상적으로 작동합니다.



## 🛠 주요 기능 (Features)
- **성능 비교** : 랜덤으로 생성된 데이터로 학습한 ANN과 실제 산에서 측정한 데이터셋으로 학습한 ANN과의 성능 비교

### 환경 설정
```bash
pip install gymnasium torch numpy matplotlib pandas
```

### 실행 방법
1. 상단의 링크에서 데이터셋을 다운로드합니다. 같은 폴더 안에 넣습니다.
2. ann_trian.py와 ann_train_dataset ver를 학습 시킵니다.
3. compare_ann_models.py로 성능을 비교해 봅니다.
python train.py
```

### 💡 팁
1. **이미지 추가**: 나중에 시뮬레이션 결과(드론이 신호를 찾아가는 경로 그래프 등)를 캡처해서 깃허브 폴더에 올린 뒤, README 중간에 넣으면 훨씬 전문적인 저장소가 됩니다.
2. **파일명 확인**: `train.py`나 `rssi_env.py` 등 실제 파일 이름이 위 내용과 다르다면 해당 부분만 수정해 주세요.
