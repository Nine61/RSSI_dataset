import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from read_dataset import load_all_avalanche_data

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "distance_ann_model.pth"
NORM_PATH = BASE_DIR / "distance_ann_norm.npz"

class DistanceANN(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=2):
        super().__init__()
        # 입력: [RSSI, SNR, 방위각, 앙각, 드론경도, 드론위도, 드론고도]
        # 출력: [조난자경도, 조난자위도]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

def train_ann(
    # 💡 데이터가 있는 실제 절대경로를 여기에 넣으세요!
    base_directory=BASE_DIR / "dataset",
    epochs=150,
    batch_size=256,
    lr=1e-3,
    model_path=MODEL_PATH,
    norm_path=NORM_PATH
):
    print("🚀 실제 산악 데이터 기반 ANN 학습을 시작합니다...")
    
    # 1. read_dataset에서 병합된 진짜 데이터 가져오기
    X_sensor, Y_target, df = load_all_avalanche_data(base_directory)
    
    if df is None:
        print("❌ 데이터를 불러오지 못했습니다. 경로를 확인해주세요.")
        return

    # 2. X_full 만들기: 센서 데이터(X_sensor) 옆에 드론 위치를 이어 붙임
    drone_positions = df[['longitude', 'latitude', 'height[m]']].values
    X_full = np.hstack([X_sensor, drone_positions]) 
    Y_full = Y_target
    
    # 3. 데이터 섞기
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X_full))
    X_full, Y_full = X_full[perm], Y_full[perm]
    
    # 4. Train / Val 나누기
    split = int(len(X_full) * 0.85)
    X_train, X_val = X_full[:split], X_full[split:]
    Y_train, Y_val = Y_full[:split], Y_full[split:]

    # 5. 정규화 (데이터 스케일 맞추기)
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-6

    X_train_n = (X_train - x_mean) / x_std
    X_val_n = (X_val - x_mean) / x_std

    model = DistanceANN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train_n, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_n, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)

    best_val = float("inf")
    best_state = None

    print(f"ANN 모델 저장 위치: {model_path}")
    
    # 6. 학습 시작
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train_t))
        
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train_t[idx]
            yb = Y_train_t[idx]

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, Y_val_t).item()
            val_mae = torch.mean(torch.abs(val_pred - Y_val_t)).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"[Epoch {epoch:03d}] 검증 손실(MSE)={val_loss:.6f} | 좌표 오차(MAE)={val_mae:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), model_path)
    np.savez(norm_path, x_mean=x_mean.astype(np.float32), x_std=x_std.astype(np.float32))
    print(f"🎉 성공! ANN 저장 완료: {model_path}")


def load_ann(model_path=MODEL_PATH, norm_path=NORM_PATH):
    model = DistanceANN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    norm = np.load(norm_path)
    x_mean = norm["x_mean"].astype(np.float32)
    x_std = norm["x_std"].astype(np.float32)
    return model, x_mean, x_std

if __name__ == "__main__":
    train_ann()