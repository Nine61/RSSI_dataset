import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from env import DroneEnv, MAP_SIZE, OBS_RADIUS, BPSK_SIGNAL_RADIUS
from ppo import PPO
from follow import (
    FOLLOW_DELAY_1,
    FOLLOW_DELAY_2,
    build_delayed_trajectory,
)
from astar import (
    safe_path_trajectory,
    pad_to_length,
)
from formation_optimizer import find_best_formation
# 삼변측량은 이제 필요 없습니다! 
from ann_train import load_ann

BASE_DIR = Path(__file__).resolve().parent
PPO_MODEL_PATH = BASE_DIR / "best_ppo_drone.pth"
REWARDS_PATH = BASE_DIR / "rewards_history.npy"
# 💡 새롭게 학습된 실제 데이터 기반 ANN 모델 경로
ANN_MODEL_PATH = BASE_DIR / "distance_ann_model.pth"
ANN_NORM_PATH = BASE_DIR / "distance_ann_norm.npz"

FORMATION_HOLD_STEPS = 20
ANIMATION_INTERVAL = 40

def assign_paths_min_total(env, current_positions, target_points):
    from itertools import permutations

    best_perm = None
    best_paths = None
    best_cost = None

    for perm in permutations(range(3)):
        paths = []
        total_cost = 0

        for drone_idx, target_idx in enumerate(perm):
            traj = safe_path_trajectory(env, current_positions[drone_idx], target_points[target_idx])
            paths.append(traj)
            total_cost += len(traj)

        if best_cost is None or total_cost < best_cost:
            best_cost = total_cost
            best_perm = perm
            best_paths = paths

    return best_perm, best_paths

def generate_synthetic_sensor_data(drone_pos, target_pos):
    """
    시뮬레이션 화면 좌표계(0~600)를 실제 데이터(위경도) 스케일로 가짜 변환하여
    ANN 모델이 이해할 수 있는 입력 [RSSI, SNR, 방위각, 앙각, 드론경도, 드론위도, 고도]를 만듭니다.
    """
    # 1. 간단한 RSSI, SNR 모사 (거리에 반비례)
    dist = np.linalg.norm(target_pos - drone_pos)
    fake_rssi = -50 - (dist / BPSK_SIGNAL_RADIUS) * 50  # -50 ~ -100 dBm
    fake_snr = 15 - (dist / BPSK_SIGNAL_RADIUS) * 20    # 15 ~ -5 dB
    
    # 2. 좌표 변환 모사 (시뮬레이션 X,Y -> 가짜 위경도)
    base_lat, base_lon = 46.37, 11.82 
    fake_drone_lat = base_lat + (drone_pos[1] / 111000)
    fake_drone_lon = base_lon + (drone_pos[0] / (111000 * math.cos(math.radians(base_lat))))
    fake_drone_alt = 1900.0 # 드론 비행 고도
    
    fake_target_lat = base_lat + (target_pos[1] / 111000)
    fake_target_lon = base_lon + (target_pos[0] / (111000 * math.cos(math.radians(base_lat))))
    fake_target_alt = 1870.0 # 조난자 고도
    
    # 3. 방위각, 앙각 계산
    dx = (fake_target_lon - fake_drone_lon) * 111000 * math.cos(math.radians(fake_target_lat))
    dy = (fake_target_lat - fake_drone_lat) * 111000
    dz = fake_target_alt - fake_drone_alt
    
    fake_azimuth = math.degrees(math.atan2(dy, dx))
    fake_elevation = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    
    # 7개의 입력 특성 반환
    return np.array([fake_rssi, fake_snr, fake_azimuth, fake_elevation, 
                     fake_drone_lon, fake_drone_lat, fake_drone_alt], dtype=np.float32)

def evaluate_and_animate():
    env = DroneEnv()
    agent = PPO(env.state_dim, env.action_dim)

    try:
        agent.policy.load_state_dict(torch.load(PPO_MODEL_PATH, weights_only=True))
        rewards_history = np.load(REWARDS_PATH)
        # 💡 새로운 다이렉트 예측 모델 로드
        ann_model, ann_mean, ann_std = load_ann(ANN_MODEL_PATH, ANN_NORM_PATH)
    except FileNotFoundError as e:
        print(f"에러: 필요한 파일이 없습니다 -> {e}")
        return

    state = env.reset()
    leader_base = [env.drone_pos.copy()]

    final_info = ""
    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mu = agent.policy.actor(state_tensor)
        action = mu.numpy()[0]

        state, reward, done, info = env.step(action)
        leader_base.append(env.drone_pos.copy())

        if done:
            final_info = info
            print(f"테스트 비행 종료: {info} (총 스텝: {len(leader_base)})")
            break

    leader_base = np.array(leader_base, dtype=float)

    phase1_total_frames = len(leader_base) + FOLLOW_DELAY_2
    leader_traj = build_delayed_trajectory(leader_base, phase1_total_frames, 0)
    follower1_traj = build_delayed_trajectory(leader_base, phase1_total_frames, FOLLOW_DELAY_1)
    follower2_traj = build_delayed_trajectory(leader_base, phase1_total_frames, FOLLOW_DELAY_2)

    best_points = None
    est_pos_sim = None
    summary_text = ""

    if final_info == "Goal Reached":
        current_positions = np.array([
            leader_traj[-1], follower1_traj[-1], follower2_traj[-1]
        ], dtype=np.float32)

        best = find_best_formation(target_pos=env.bpsk_pos.astype(np.float32), obstacles=env.obstacles)
        best_points = best["best_points"]
        
        assignment_perm, paths = assign_paths_min_total(env, current_positions, best_points)
        trans_len = max(len(paths[0]), len(paths[1]), len(paths[2]), 1)

        leader_path = pad_to_length(paths[0], trans_len)
        follower1_path = pad_to_length(paths[1], trans_len)
        follower2_path = pad_to_length(paths[2], trans_len)

        leader_traj = np.vstack([leader_traj, leader_path])
        follower1_traj = np.vstack([follower1_traj, follower1_path])
        follower2_traj = np.vstack([follower2_traj, follower2_path])

        leader_hold = np.repeat(leader_traj[-1][None, :], FORMATION_HOLD_STEPS, axis=0)
        follower1_hold = np.repeat(follower1_traj[-1][None, :], FORMATION_HOLD_STEPS, axis=0)
        follower2_hold = np.repeat(follower2_traj[-1][None, :], FORMATION_HOLD_STEPS, axis=0)

        leader_traj = np.vstack([leader_traj, leader_hold])
        follower1_traj = np.vstack([follower1_traj, follower1_hold])
        follower2_traj = np.vstack([follower2_traj, follower2_hold])

        # 💡 삼변측량 없이, 모델이 직접 좌표를 예측합니다!
        # 리더 드론이 측정한 데이터라고 가정
        sensor_data = generate_synthetic_sensor_data(leader_traj[-1], env.bpsk_pos)
        
        # 정규화
        feat_n = (sensor_data - ann_mean) / ann_std
        with torch.no_grad():
            pred_geo = ann_model(torch.tensor(feat_n[None, :], dtype=torch.float32)).numpy()[0]
        
        # 모델은 [경도, 위도]를 뱉어내므로, 다시 시뮬레이션 X, Y로 역변환하여 그려줍니다.
        base_lat, base_lon = 46.37, 11.82 
        est_y = (pred_geo[1] - base_lat) * 111000
        est_x = (pred_geo[0] - base_lon) * (111000 * math.cos(math.radians(base_lat)))
        est_pos_sim = np.array([est_x, est_y])
        
        pos_err = np.linalg.norm(est_pos_sim - env.bpsk_pos)

        print("\n=== AI 다이렉트 좌표 예측 결과 ===")
        print(f"실제 시뮬 Target : ({env.bpsk_pos[0]:.2f}, {env.bpsk_pos[1]:.2f})")
        print(f"AI 추정 Target   : ({est_pos_sim[0]:.2f}, {est_pos_sim[1]:.2f})")
        print(f"위치 오차        : {pos_err:.4f} m")

        summary_text = (
            f"AI Direct Prediction\n"
            f"True = ({env.bpsk_pos[0]:.1f}, {env.bpsk_pos[1]:.1f})\n"
            f"Est  = ({est_pos_sim[0]:.1f}, {est_pos_sim[1]:.1f})\n"
            f"Err  = {pos_err:.2f} m"
        )

    # 시각화 부분 (이전과 동일하게 Matplotlib 활용)
    total_frames = len(leader_traj)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.set_xlim(0, MAP_SIZE)
    ax1.set_ylim(0, MAP_SIZE)
    ax1.set_title("Drone Tracking + Direct AI Localization")
    ax1.set_aspect("equal")

    signal_circle = plt.Circle(env.bpsk_pos, BPSK_SIGNAL_RADIUS, color="red", alpha=0.2, label="Signal Radius")
    ax1.add_patch(signal_circle)
    ax1.plot(env.bpsk_pos[0], env.bpsk_pos[1], "ro", markersize=6, label="Target")

    for obs in env.obstacles:
        obs_circle = plt.Circle(obs, OBS_RADIUS, color="yellow", ec="black")
        ax1.add_patch(obs_circle)

    leader_dot, = ax1.plot([], [], "o", color="purple", markersize=8, label="Leader")
    follower1_dot, = ax1.plot([], [], "o", color="green", markersize=8, label="Follower 1")
    follower2_dot, = ax1.plot([], [], "o", color="blue", markersize=8, label="Follower 2")

    leader_path_line, = ax1.plot([], [], "-", color="purple", alpha=0.4)
    follower1_path_line, = ax1.plot([], [], "--", color="green", alpha=0.4)
    follower2_path_line, = ax1.plot([], [], "--", color="blue", alpha=0.4)

    est_marker = None
    if est_pos_sim is not None:
        est_marker, = ax1.plot([est_pos_sim[0]], [est_pos_sim[1]], "x", color="black", markersize=10, mew=2, label="Estimated Target")

    phase_text = ax1.text(0.02, 0.98, "", transform=ax1.transAxes, va="top", fontsize=10,
                          bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    if summary_text:
        ax1.text(1.02, 0.98, summary_text, transform=ax1.transAxes, va="top", ha="left", fontsize=9,
                 bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.95))

    ax1.legend(loc="upper right")

    # 보상 그래프 (Matplotlib 활용)
    try:
        ax2.plot(np.load(REWARDS_PATH), color="blue", alpha=0.7)
    except Exception:
        pass
    ax2.set_title("Episode vs Training Reward")
    ax2.grid(True)

    def update(frame):
        leader_dot.set_data([leader_traj[frame, 0]], [leader_traj[frame, 1]])
        leader_path_line.set_data(leader_traj[:frame + 1, 0], leader_traj[:frame + 1, 1])

        follower1_dot.set_data([follower1_traj[frame, 0]], [follower1_traj[frame, 1]])
        follower1_path_line.set_data(follower1_traj[:frame + 1, 0], follower1_traj[:frame + 1, 1])

        follower2_dot.set_data([follower2_traj[frame, 0]], [follower2_traj[frame, 1]])
        follower2_path_line.set_data(follower2_traj[:frame + 1, 0], follower2_traj[:frame + 1, 1])

        artists = [leader_dot, follower1_dot, follower2_dot, leader_path_line, follower1_path_line, follower2_path_line, phase_text]
        if est_marker is not None and frame >= total_frames - FORMATION_HOLD_STEPS:
            artists.append(est_marker)
        return tuple(artists)

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=ANIMATION_INTERVAL, blit=True, repeat=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_and_animate()