import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from env import MAP_SIZE, BPSK_SIGNAL_RADIUS
from rssi_env import RealisticRSSIEnv
from kalman_filter import filter_three_links, extract_mean_std_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "orig_ann_model.pth"
NORM_PATH = BASE_DIR / "orig_ann_norm.npz"


class DistanceANN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_noisy_triangle_anchors_around_target(target_pos, base_radius, base_theta, rng,
                                              radius_jitter=12.0, angle_jitter_deg=12.0):
    """
    타겟 주변에 3개 드론 위치를 만든다.
    완전 정삼각형만 쓰지 않고, 각 꼭짓점의 반경과 각도를 조금씩 흔들어
    약간 어긋난 삼각형도 학습에 포함시킨다.
    """
    anchors = []
    angle_jitter_rad = math.radians(angle_jitter_deg)

    for k in range(3):
        base_ang = base_theta + (2.0 * math.pi / 3.0) * k
        radius = base_radius + rng.uniform(-radius_jitter, radius_jitter)
        radius = max(30.0, radius)
        ang = base_ang + rng.uniform(-angle_jitter_rad, angle_jitter_rad)
        anchors.append([
            target_pos[0] + radius * math.cos(ang),
            target_pos[1] + radius * math.sin(ang),
        ])

    return np.array(anchors, dtype=np.float32)


def generate_dataset(n_samples=12000, k_rssi_samples=20, seed=42):
    rng = np.random.default_rng(seed)
    rssi_sim = RealisticRSSIEnv(seed=seed)

    obstacles = [
        np.array([200.0, 200.0], dtype=np.float32),
        np.array([400.0, 150.0], dtype=np.float32),
        np.array([150.0, 450.0], dtype=np.float32),
        np.array([450.0, 400.0], dtype=np.float32),
    ]

    X = np.zeros((n_samples, 6), dtype=np.float32)
    Y = np.zeros((n_samples, 3), dtype=np.float32)

    for i in range(n_samples):
        target = rng.uniform(80.0, MAP_SIZE - 80.0, size=2).astype(np.float32)
        base_radius = rng.uniform(45.0, 110.0)
        base_theta = rng.uniform(0.0, 2.0 * math.pi)

        # 완전 정삼각형만 보지 않도록 약간 찌그러진 삼각형도 학습에 포함
        drone_positions = make_noisy_triangle_anchors_around_target(
            target_pos=target,
            base_radius=base_radius,
            base_theta=base_theta,
            rng=rng,
            radius_jitter=12.0,
            angle_jitter_deg=12.0,
        )

        d_true = np.linalg.norm(drone_positions - target[None, :], axis=1).astype(np.float32)

        rssi_samples = rssi_sim.sample_three_links(
            drone_positions=drone_positions,
            target_pos=target,
            obstacles=obstacles,
            k_samples=k_rssi_samples,
        )

        # ANN 입력은 raw RSSI가 아니라 칼만필터를 거친 값의 평균/표준편차
        filtered = filter_three_links(rssi_samples)
        feat = extract_mean_std_features(filtered)

        X[i] = feat
        Y[i] = np.clip(d_true, 1.0, BPSK_SIGNAL_RADIUS)

    return X, Y


def train_ann(
    model_path=MODEL_PATH,
    norm_path=NORM_PATH,
    n_samples=12000,
    k_rssi_samples=20,
    epochs=150,
    batch_size=256,
    lr=1e-3,
    seed=42,
):
    X, Y = generate_dataset(
        n_samples=n_samples,
        k_rssi_samples=k_rssi_samples,
        seed=seed,
    )

    # 데이터 생성 순서 영향 줄이기 위해 한번 섞고 나눈다
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X = X[perm]
    Y = Y[perm]

    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

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
    print(f"ANN 정규화 저장 위치: {norm_path}")

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

        if epoch % 20 == 0:
            print(f"[Epoch {epoch:03d}] val_mse={val_loss:.4f}  val_mae={val_mae:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), model_path)
    np.savez(norm_path, x_mean=x_mean.astype(np.float32), x_std=x_std.astype(np.float32))
    print(f"ANN 저장 완료: {model_path}, {norm_path}")


def load_ann(model_path=MODEL_PATH, norm_path=NORM_PATH):
    model = DistanceANN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    norm = np.load(norm_path)
    x_mean = norm["x_mean"].astype(np.float32)
    x_std = norm["x_std"].astype(np.float32)
    return model, x_mean, x_std


def predict_distances(filtered_rssi_3xk, model, x_mean, x_std):
    feat = extract_mean_std_features(filtered_rssi_3xk)
    feat_n = (feat - x_mean) / x_std
    with torch.no_grad():
        pred = model(torch.tensor(feat_n[None, :], dtype=torch.float32))
    d_pred = pred.numpy().reshape(-1).astype(np.float32)
    return np.clip(d_pred, 1.0, BPSK_SIGNAL_RADIUS)


if __name__ == "__main__":
    train_ann()
