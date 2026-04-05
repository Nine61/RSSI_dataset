import numpy as np

FOLLOW_DELAY_1 = 12
FOLLOW_DELAY_2 = 24


def delayed_position(trajectory, frame, delay):
    # delay만큼 이전 위치를 따라감
    idx = frame - delay
    if idx < 0:
        idx = 0
    if idx >= len(trajectory):
        idx = len(trajectory) - 1
    return trajectory[idx].copy()


def build_delayed_trajectory(trajectory, total_frames, delay):
    # follower의 전체 지연 경로 생성
    return np.array(
        [delayed_position(trajectory, f, delay) for f in range(total_frames)],
        dtype=float
    )


def rotate_vector(vec, deg):
    # 2차원 벡터를 deg도 회전
    rad = np.deg2rad(deg)
    c = np.cos(rad)
    s = np.sin(rad)
    return np.array([
        c * vec[0] - s * vec[1],
        s * vec[0] + c * vec[1]
    ], dtype=float)
