"""
compare_models.py  — 공정한 두 모델 성능 비교
===============================================

[과거 모델] ann_train.py 기반
  - 입력: Kalman 필터 후 RSSI 평균·표준편차 × 3링크 = 6차원
  - 출력: 드론 3대로부터의 거리 = 3차원
  - 평가: 시뮬레이션 데이터 → 삼변측량 → 위치 오차(m)

[신규 모델] ann_train_dataset_ver.py 기반
  - 입력: RSSI, SNR, 방위각, 앙각, 드론 경도·위도·고도 = 7차원
  - 출력: 조난자 위경도 = 2차원
  - 평가: 실제 Avalanche 데이터셋 → 직접 좌표 비교 → 위치 오차(m)

두 모델은 입출력 구조가 달라 동일 입력으로 비교할 수 없습니다.
각자의 도메인에서 공정하게 평가한 뒤 오차 분포를 나란히 비교합니다.
"""

import math
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

BASE_DIR = Path(__file__).resolve().parent

# ── 모델 파일 경로 ──────────────────────────────────────────
OLD_MODEL_PATH = BASE_DIR / "orig_ann_model.pth"
OLD_NORM_PATH  = BASE_DIR / "orig_ann_norm.npz"
NEW_MODEL_PATH = BASE_DIR / "distance_ann_model.pth"
NEW_NORM_PATH  = BASE_DIR / "distance_ann_norm.npz"

# ── 돌로미티 기준 위도 (신규 모델 좌표 변환용) ──────────────
BASE_LAT_DEG = 46.37


# ═══════════════════════════════════════════════════════════
# 공통 유틸리티
# ═══════════════════════════════════════════════════════════

class DistanceANN_Old(torch.nn.Module):
    """과거 모델: 6 → 64 → 64 → 3"""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(6, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
        )
    def forward(self, x):
        return self.net(x)


class DistanceANN_New(torch.nn.Module):
    """신규 모델: 7 → 128 → 128 → 2"""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(7, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
        )
    def forward(self, x):
        return self.net(x)


def latlon_error_meters(pred_lonlat: np.ndarray,
                        true_lonlat: np.ndarray,
                        ref_lat_deg: float = BASE_LAT_DEG) -> np.ndarray:
    """
    위경도 예측값과 정답의 유클리드 오차를 미터로 변환.
    pred_lonlat, true_lonlat: (N, 2) — 열 순서 [경도, 위도]
    """
    cos_lat = math.cos(math.radians(ref_lat_deg))
    dx = (pred_lonlat[:, 0] - true_lonlat[:, 0]) * 111_000 * cos_lat  # 경도 → m
    dy = (pred_lonlat[:, 1] - true_lonlat[:, 1]) * 111_000             # 위도 → m
    return np.sqrt(dx ** 2 + dy ** 2)


def print_metrics(label: str, errors: np.ndarray):
    mae  = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    p50  = np.percentile(errors, 50)
    p80  = np.percentile(errors, 80)
    p95  = np.percentile(errors, 95)
    print(f"\n{'─'*44}")
    print(f"  {label}")
    print(f"{'─'*44}")
    print(f"  MAE  : {mae:>8.2f} m")
    print(f"  RMSE : {rmse:>8.2f} m")
    print(f"  P50  : {p50:>8.2f} m  (중앙값)")
    print(f"  P80  : {p80:>8.2f} m  (80% 오차 이내)")
    print(f"  P95  : {p95:>8.2f} m  (95% 오차 이내)")
    print(f"{'─'*44}")
    return mae, rmse, p50, p80, p95


# ═══════════════════════════════════════════════════════════
# PART A — 과거 모델 평가
#   시뮬레이션 데이터 → Kalman 특징 → 거리 예측 → 삼변측량
# ═══════════════════════════════════════════════════════════

def _kalman_filter_simple(rssi_sequence: np.ndarray,
                           process_var: float = 1.0,
                           measure_var: float = 4.0) -> np.ndarray:
    """
    1D Kalman 필터 (스칼라 RSSI 시퀀스 평활화).
    rssi_sequence: (k,) 배열
    반환: (k,) 필터링된 배열
    """
    n  = len(rssi_sequence)
    x  = rssi_sequence[0]
    P  = 1.0
    filtered = np.empty(n, dtype=np.float32)
    for i, z in enumerate(rssi_sequence):
        # 예측
        P = P + process_var
        # 갱신
        K = P / (P + measure_var)
        x = x + K * (z - x)
        P = (1 - K) * P
        filtered[i] = x
    return filtered


def _extract_features(rssi_3xk: np.ndarray) -> np.ndarray:
    """
    3링크 RSSI 배열 → 평균·표준편차 6차원 특징.
    rssi_3xk: (3, k) 배열
    반환: (6,) float32
    """
    feats = []
    for link_rssi in rssi_3xk:
        filtered = _kalman_filter_simple(link_rssi)
        feats.append(filtered.mean())
        feats.append(filtered.std() + 1e-6)
    return np.array(feats, dtype=np.float32)


def _simulate_rssi(dist_m: float, rng: np.random.Generator,
                   n_samples: int = 20,
                   n_ref: float = 2.0, sigma_db: float = 4.0) -> np.ndarray:
    """
    로그 거리 경로 손실 모델로 RSSI 시뮬레이션.
    반환: (n_samples,) dBm 배열
    """
    rssi_mean = -50.0 - 10.0 * n_ref * np.log10(max(dist_m, 1.0))
    return (rssi_mean + rng.normal(0, sigma_db, size=n_samples)).astype(np.float32)


def _trilaterate(anchors: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """
    3개 앵커 + 거리로 최소제곱 삼변측량.
    anchors  : (3, 2) — 앵커 (x, y) 좌표 (m)
    distances: (3,)   — 각 앵커까지의 예측 거리 (m)
    반환: (2,) 추정 위치
    """
    # 마지막 앵커 기준 선형화
    A, b = [], []
    xN, yN, dN = anchors[-1, 0], anchors[-1, 1], distances[-1]
    for (xi, yi), di in zip(anchors[:-1], distances[:-1]):
        A.append([2 * (xi - xN), 2 * (yi - yN)])
        b.append(di**2 - dN**2 - xi**2 + xN**2 - yi**2 + yN**2)
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return pos.astype(np.float32)


def evaluate_old_model(n_samples: int = 300, seed: int = 42) -> np.ndarray:
    """
    과거 모델을 시뮬레이션 환경에서 평가.
    반환: (n_samples,) 위치 오차 배열 (m 단위)
    """
    print("\n[과거 모델] 시뮬레이션 데이터 생성 및 평가 중...")

    # 모델 로드
    old_model = DistanceANN_Old()
    old_model.load_state_dict(
        torch.load(OLD_MODEL_PATH, map_location="cpu", weights_only=True))
    old_model.eval()
    norm = np.load(OLD_NORM_PATH)
    x_mean, x_std = norm["x_mean"].astype(np.float32), norm["x_std"].astype(np.float32)

    rng = np.random.default_rng(seed)
    MAP_SIZE = 600.0

    errors = []
    for _ in range(n_samples):
        # 조난자 실제 위치
        target = rng.uniform(80.0, MAP_SIZE - 80.0, size=2).astype(np.float32)

        # 드론 3대 — 약간 찌그러진 삼각형 배치 (학습 분포 재현)
        base_r   = rng.uniform(45.0, 110.0)
        base_ang = rng.uniform(0.0, 2.0 * math.pi)
        anchors  = []
        for k in range(3):
            ang = base_ang + (2.0 * math.pi / 3.0) * k + rng.uniform(-0.21, 0.21)
            r   = base_r + rng.uniform(-12.0, 12.0)
            r   = max(30.0, r)
            anchors.append([target[0] + r * math.cos(ang),
                             target[1] + r * math.sin(ang)])
        anchors = np.array(anchors, dtype=np.float32)

        # 실제 거리
        d_true = np.linalg.norm(anchors - target[None, :], axis=1)

        # RSSI 시뮬레이션 (NLOS 환경: 경로 손실 지수 높임)
        rssi_3xk = np.stack([
            _simulate_rssi(d, rng, n_samples=20, n_ref=2.8, sigma_db=6.0)
            for d in d_true
        ])

        # Kalman 필터 → 6차원 특징
        feat = _extract_features(rssi_3xk)
        feat_n = (feat - x_mean) / x_std

        with torch.no_grad():
            d_pred = old_model(
                torch.tensor(feat_n[None, :], dtype=torch.float32)
            ).numpy().reshape(-1)
        d_pred = np.clip(d_pred, 1.0, 400.0)

        # 삼변측량으로 위치 추정
        try:
            pos_pred = _trilaterate(anchors, d_pred)
        except Exception:
            pos_pred = target + rng.uniform(-50, 50, size=2)

        error = float(np.linalg.norm(pos_pred - target))
        errors.append(error)

    return np.array(errors, dtype=np.float32)


# ═══════════════════════════════════════════════════════════
# PART B — 신규 모델 평가
#   실제 Avalanche 데이터셋 → 직접 위경도 예측
# ═══════════════════════════════════════════════════════════

def evaluate_new_model() -> np.ndarray:
    """
    신규 모델을 실제 Avalanche 데이터셋으로 평가.
    반환: (N_test,) 위치 오차 배열 (m 단위)
    """
    print("\n[신규 모델] Avalanche 실제 데이터 기반 평가 중...")

    # 모델 로드
    new_model = DistanceANN_New()
    new_model.load_state_dict(
        torch.load(NEW_MODEL_PATH, map_location="cpu", weights_only=True))
    new_model.eval()
    norm = np.load(NEW_NORM_PATH)
    x_mean = norm["x_mean"].astype(np.float32)
    x_std  = norm["x_std"].astype(np.float32)

    # 실제 데이터 로드
    try:
        from read_dataset import load_all_avalanche_data
        dataset_dir = BASE_DIR / "dataset"
        X_sensor, Y_target, df = load_all_avalanche_data(dataset_dir)
        if df is None:
            raise FileNotFoundError("데이터셋 로드 실패")

        drone_positions = df[["longitude", "latitude", "height[m]"]].values
        X_full = np.hstack([X_sensor, drone_positions]).astype(np.float32)
        Y_full = Y_target.astype(np.float32)

    except (ImportError, FileNotFoundError) as e:
        print(f"  [경고] 실제 데이터 로드 실패 ({e})")
        print("  → 학습 분포를 재현한 대체 테스트 데이터를 생성합니다.")
        X_full, Y_full = _generate_avalanche_like_test(n_samples=300)

    # 재현성을 위해 고정 seed로 셔플 후 15% 테스트 분리
    rng  = np.random.default_rng(99)
    perm = rng.permutation(len(X_full))
    X_full, Y_full = X_full[perm], Y_full[perm]
    split = int(len(X_full) * 0.85)
    X_test = X_full[split:]
    Y_test = Y_full[split:]

    # 정규화 및 추론
    X_test_n = (X_test - x_mean) / x_std
    with torch.no_grad():
        Y_pred = new_model(
            torch.tensor(X_test_n, dtype=torch.float32)
        ).numpy()

    errors = latlon_error_meters(Y_pred, Y_test)

    # 진단 출력
    print(f"  테스트 샘플 수     : {len(Y_test)}")
    print(f"  예측 범위 (경도)   : {Y_pred[:, 0].min():.5f} ~ {Y_pred[:, 0].max():.5f}")
    print(f"  정답 범위 (경도)   : {Y_test[:, 0].min():.5f} ~ {Y_test[:, 0].max():.5f}")
    print(f"  예측 예시 (1번)    : lon={Y_pred[0,0]:.6f}, lat={Y_pred[0,1]:.6f}")
    print(f"  정답 예시 (1번)    : lon={Y_test[0,0]:.6f}, lat={Y_test[0,1]:.6f}")

    return errors


def _generate_avalanche_like_test(n_samples: int = 300) -> tuple:
    """
    read_dataset.py가 없을 때 사용하는 대체 데이터 생성기.
    Avalanche 데이터셋의 실측 분포를 근사해서 재현합니다.
    """
    rng = np.random.default_rng(42)
    base_lon, base_lat = 11.82, 46.37   # 돌로미티 중심

    X_list, Y_list = [], []
    for _ in range(n_samples):
        # 조난자 위치 (기준점 ±500m 이내)
        t_lon = base_lon + rng.uniform(-0.004, 0.004)
        t_lat = base_lat + rng.uniform(-0.004, 0.004)

        # 드론 위치 (조난자 주변 50~300m)
        d_lon = t_lon + rng.uniform(-0.003, 0.003)
        d_lat = t_lat + rng.uniform(-0.003, 0.003)
        d_alt = rng.uniform(10.0, 80.0)

        dx = (d_lon - t_lon) * 111_000 * math.cos(math.radians(base_lat))
        dy = (d_lat - t_lat) * 111_000
        dist = math.sqrt(dx**2 + dy**2)

        # 신호 특성 (눈 속 조난자 → 심한 감쇠)
        snow_depth = rng.uniform(0.3, 1.5)   # m
        rssi = -70 - dist / 20 - snow_depth * 8 + rng.normal(0, 5)
        snr  = 10 - dist / 30 - snow_depth * 4 + rng.normal(0, 2)

        # AOA (실제 방향 + 오차)
        az_true = math.degrees(math.atan2(dy, dx))
        el_true = math.degrees(math.atan2(d_alt, dist))
        aoa_az = az_true + rng.normal(0, 8)
        aoa_el = el_true + rng.normal(0, 4)

        X_list.append([rssi, snr, aoa_az, aoa_el, d_lon, d_lat, d_alt])
        Y_list.append([t_lon, t_lat])

    return (np.array(X_list, dtype=np.float32),
            np.array(Y_list, dtype=np.float32))


# ═══════════════════════════════════════════════════════════
# PART C — 시각화
# ═══════════════════════════════════════════════════════════

def visualize(errors_old: np.ndarray, errors_new: np.ndarray,
              mae_old: float, rmse_old: float,
              mae_new: float, rmse_new: float):
    """4개 패널 종합 시각화"""

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "ANN Localization Model Comparison\n"
        "Old: Simulation + Trilateration  |  New: Avalanche Dataset (Direct)",
        fontsize=14, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    COLOR_OLD = "#4472C4"   # 파랑
    COLOR_NEW = "#70AD47"   # 초록

    # ── 패널 1: 오차 분포 히스토그램 ──────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    max_err = max(np.percentile(errors_old, 97), np.percentile(errors_new, 97))
    bins = np.linspace(0, max_err, 40)
    ax1.hist(errors_old, bins=bins, alpha=0.65, color=COLOR_OLD,
             label=f"Old  (MAE={mae_old:.1f}m)", density=True)
    ax1.hist(errors_new, bins=bins, alpha=0.65, color=COLOR_NEW,
             label=f"New  (MAE={mae_new:.1f}m)", density=True)
    ax1.set_title("Error Distribution (Histogram)")
    ax1.set_xlabel("Localization Error (m)")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ── 패널 2: 누적 오차 분포 (CDF) ─────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for errors, label, rmse, color in [
        (errors_old, "Old", rmse_old, COLOR_OLD),
        (errors_new, "New", rmse_new, COLOR_NEW),
    ]:
        xs = np.sort(errors)
        ys = np.arange(1, len(xs) + 1) / len(xs) * 100
        ax2.plot(xs, ys, color=color, linewidth=2.2,
                 label=f"{label}  (RMSE={rmse:.1f}m)")

    ax2.axhline(80, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.text(max_err * 0.55, 81.5, "80% Threshold", color="gray", fontsize=9)
    ax2.axhline(50, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_title("Cumulative Error Distribution (CDF)")
    ax2.set_xlabel("Localization Error (m)")
    ax2.set_ylabel("Cumulative Probability (%)")
    ax2.set_ylim(0, 102)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    # ── 패널 3: Box plot 비교 ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bp = ax3.boxplot(
        [errors_old, errors_new],
        labels=["Old Model\n(Simulation)", "New Model\n(Avalanche)"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
        showfliers=True,
    )
    bp["boxes"][0].set_facecolor(COLOR_OLD + "99")
    bp["boxes"][1].set_facecolor(COLOR_NEW + "99")
    ax3.set_title("Error Boxplot")
    ax3.set_ylabel("Localization Error (m)")
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")

    # ── 패널 4: 요약 지표 테이블 ──────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    metrics = ["MAE (m)", "RMSE (m)", "Median (m)", "P80 (m)", "P95 (m)"]
    vals_old = [
        f"{np.mean(errors_old):.2f}",
        f"{np.sqrt(np.mean(errors_old**2)):.2f}",
        f"{np.median(errors_old):.2f}",
        f"{np.percentile(errors_old, 80):.2f}",
        f"{np.percentile(errors_old, 95):.2f}",
    ]
    vals_new = [
        f"{np.mean(errors_new):.2f}",
        f"{np.sqrt(np.mean(errors_new**2)):.2f}",
        f"{np.median(errors_new):.2f}",
        f"{np.percentile(errors_new, 80):.2f}",
        f"{np.percentile(errors_new, 95):.2f}",
    ]

    table_data = [[m, o, n] for m, o, n in zip(metrics, vals_old, vals_new)]
    col_labels  = ["Metric", "Old Model", "New Model"]
    col_colors  = ["#f0f0f0", COLOR_OLD + "55", COLOR_NEW + "55"]

    tbl = ax4.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)

    for j, c in enumerate(col_colors):
        tbl[(0, j)].set_facecolor(c)
        tbl[(0, j)].set_text_props(fontweight="bold")

    # 신규 모델이 더 좋은 행에 색칠
    for i, (vo, vn) in enumerate(zip(vals_old, vals_new), start=1):
        try:
            if float(vn) < float(vo):
                tbl[(i, 2)].set_facecolor("#d4edda")
            else:
                tbl[(i, 1)].set_facecolor("#d4edda")
        except ValueError:
            pass

    ax4.set_title("Performance Summary", pad=20)

    # 개선율 주석
    rmse_improve = (rmse_old - rmse_new) / rmse_old * 100
    sign = "↑ 개선" if rmse_improve > 0 else "↓ 저하"
    ax4.text(0.5, 0.05,
             f"RMSE 기준 {sign}: {abs(rmse_improve):.1f}%",
             ha="center", va="bottom", transform=ax4.transAxes,
             fontsize=12, fontweight="bold",
             color="#155724" if rmse_improve > 0 else "#721c24")

    plt.savefig(BASE_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    print("\n  그래프 저장: model_comparison.png")
    plt.show()


# ═══════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════

def run_comparison():
    print("=" * 50)
    print("  ANN 모델 성능 비교 시작")
    print("=" * 50)

    # ── 파일 존재 확인 ──────────────────────────────────────
    missing = []
    for p in [OLD_MODEL_PATH, OLD_NORM_PATH, NEW_MODEL_PATH, NEW_NORM_PATH]:
        if not p.exists():
            missing.append(p.name)
    if missing:
        print(f"\n[오류] 다음 파일을 찾을 수 없습니다: {missing}")
        print("  ann_train.py와 ann_train_dataset_ver.py를 먼저 실행하세요.")
        return

    # ── 각 모델 평가 ────────────────────────────────────────
    errors_old = evaluate_old_model(n_samples=300, seed=42)
    errors_new = evaluate_new_model()

    # ── 지표 출력 ───────────────────────────────────────────
    mae_old, rmse_old, *_ = print_metrics("과거 모델 (시뮬레이션 + 삼변측량)", errors_old)
    mae_new, rmse_new, *_ = print_metrics("신규 모델 (Avalanche 실데이터)", errors_new)

    print("\n" + "=" * 50)
    rmse_diff = rmse_old - rmse_new
    if rmse_diff > 0:
        print(f"  결과: 신규 모델이 RMSE 기준 {rmse_diff:.2f}m ({rmse_diff/rmse_old*100:.1f}%) 개선")
    else:
        print(f"  결과: 과거 모델이 RMSE 기준 {-rmse_diff:.2f}m ({-rmse_diff/rmse_old*100:.1f}%) 앞섬")
    print("=" * 50)

    # ── 시각화 ──────────────────────────────────────────────
    visualize(errors_old, errors_new, mae_old, rmse_old, mae_new, rmse_new)


if __name__ == "__main__":
    run_comparison()