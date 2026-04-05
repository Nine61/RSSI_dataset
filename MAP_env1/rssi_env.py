import math
import numpy as np
from env import BPSK_SIGNAL_RADIUS, OBS_RADIUS


def distance_point_to_segment(point, seg_a, seg_b):
    """점과 선분 사이의 최소 거리 계산"""
    point = np.asarray(point, dtype=np.float32)
    seg_a = np.asarray(seg_a, dtype=np.float32)
    seg_b = np.asarray(seg_b, dtype=np.float32)

    ab = seg_b - seg_a
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 < 1e-9:
        return float(np.linalg.norm(point - seg_a))

    t = float(np.dot(point - seg_a, ab) / ab_len2)
    t = max(0.0, min(1.0, t))
    closest = seg_a + t * ab
    return float(np.linalg.norm(point - closest))


class RealisticRSSIEnv:
    """
    RSSI를 너무 단순한 0~1 선형 모델 대신,
    log-distance path loss + shadowing + block fade + 장애물 감쇠로 흉내내는 간단한 채널 모델.

    내부 계산은 dBm 비슷한 스케일로 하고,
    최종 출력만 0~1 범위의 정규화 신호세기로 변환한다.
    """

    def __init__(
        self,
        ref_rssi_dbm=-35.0,              # 기준 거리(d0)에서의 수신 세기
        path_loss_exponent=2.2,          # 거리 감쇠 지수
        d0=1.0,                          # 기준 거리
        shadowing_std_db=2.0,            # 링크별 고정 shadowing
        fast_fading_std_db=1.2,          # 샘플별 빠른 변동
        far_fading_gain_db=2.0,          # 멀수록 추가되는 변동
        obstacle_loss_db=6.0,            # 장애물 관통 시 감쇠
        near_obstacle_loss_db=2.5,       # 장애물 근접 시 감쇠
        fade_prob=0.08,                  # 블록 fade 시작 확률
        fade_db_range=(3.0, 8.0),        # block fade 감쇠량 범위
        fade_len_range=(2, 5),           # block fade 지속 길이
        min_rssi_dbm=-95.0,              # 정규화 하한
        max_rssi_dbm=-30.0,              # 정규화 상한
        seed=42,
    ):
        self.ref_rssi_dbm = float(ref_rssi_dbm)
        self.path_loss_exponent = float(path_loss_exponent)
        self.d0 = float(d0)

        self.shadowing_std_db = float(shadowing_std_db)
        self.fast_fading_std_db = float(fast_fading_std_db)
        self.far_fading_gain_db = float(far_fading_gain_db)

        self.obstacle_loss_db = float(obstacle_loss_db)
        self.near_obstacle_loss_db = float(near_obstacle_loss_db)

        self.fade_prob = float(fade_prob)
        self.fade_db_range = tuple(fade_db_range)
        self.fade_len_range = tuple(fade_len_range)

        self.min_rssi_dbm = float(min_rssi_dbm)
        self.max_rssi_dbm = float(max_rssi_dbm)
        self.rng = np.random.default_rng(seed)

    def ideal_rssi_dbm(self, distance):
        """log-distance path loss 기반의 이상적 RSSI(dBm 비슷한 스케일)"""
        d = max(float(distance), self.d0)
        return self.ref_rssi_dbm - 10.0 * self.path_loss_exponent * math.log10(d / self.d0)

    def obstacle_loss(self, tx_pos, rx_pos, obstacles):
        """
        직선 링크가 장애물을 관통하거나 가깝게 지나가면 감쇠 추가.
        중심을 깊게 지날수록 조금 더 크게 깎는다.
        """
        tx_pos = np.asarray(tx_pos, dtype=np.float32)
        rx_pos = np.asarray(rx_pos, dtype=np.float32)

        loss_db = 0.0
        for obs in obstacles:
            obs = np.asarray(obs, dtype=np.float32)
            seg_dist = distance_point_to_segment(obs, tx_pos, rx_pos)

            if seg_dist <= OBS_RADIUS:
                # 장애물 중심부를 더 깊게 지날수록 감쇠를 조금 더 줌
                depth_ratio = 1.0 - (seg_dist / max(OBS_RADIUS, 1e-6))
                loss_db += self.obstacle_loss_db * (1.0 + 0.5 * depth_ratio)
            elif seg_dist <= OBS_RADIUS + 20.0:
                near_ratio = 1.0 - ((seg_dist - OBS_RADIUS) / 20.0)
                loss_db += self.near_obstacle_loss_db * max(near_ratio, 0.0)

        return loss_db

    def normalize_rssi(self, rssi_dbm):
        """dBm 비슷한 값을 0~1 범위의 신호세기로 정규화"""
        x = (np.asarray(rssi_dbm, dtype=np.float32) - self.min_rssi_dbm) / (
            self.max_rssi_dbm - self.min_rssi_dbm
        )
        return np.clip(x, 0.0, 1.0).astype(np.float32)

    def sample_one_link(self, tx_pos, rx_pos, obstacles, k_samples=20, return_dbm=False):
        """
        타겟 1개 - 드론 1개 링크에 대해 RSSI를 k번 샘플링.
        기본 반환은 0~1 정규화 값.
        return_dbm=True면 내부 dBm 스케일 값을 반환.
        """
        tx_pos = np.asarray(tx_pos, dtype=np.float32)
        rx_pos = np.asarray(rx_pos, dtype=np.float32)

        distance = float(np.linalg.norm(tx_pos - rx_pos))
        ideal_dbm = self.ideal_rssi_dbm(distance)

        # 링크 전체에 공통으로 작용하는 큰 스케일 shadowing
        link_shadowing_db = self.rng.normal(0.0, self.shadowing_std_db)

        # 거리 멀수록 fast fading도 조금 더 커지게 설정
        dist_ratio = min(distance / float(BPSK_SIGNAL_RADIUS), 1.0)
        fast_std_db = self.fast_fading_std_db + self.far_fading_gain_db * dist_ratio

        # 장애물 감쇠
        obs_loss_db = self.obstacle_loss(tx_pos, rx_pos, obstacles)

        samples_dbm = np.full(k_samples, ideal_dbm + link_shadowing_db - obs_loss_db, dtype=np.float32)
        samples_dbm += self.rng.normal(0.0, fast_std_db, size=k_samples).astype(np.float32)

        # 샘플별 iid fade 대신 짧게 지속되는 block fade를 넣음
        i = 0
        while i < k_samples:
            if self.rng.random() < self.fade_prob:
                fade_len = int(self.rng.integers(self.fade_len_range[0], self.fade_len_range[1] + 1))
                fade_db = float(self.rng.uniform(self.fade_db_range[0], self.fade_db_range[1]))
                end = min(i + fade_len, k_samples)
                samples_dbm[i:end] -= fade_db
                i = end
            else:
                i += 1

        if return_dbm:
            return samples_dbm.astype(np.float32)

        return self.normalize_rssi(samples_dbm)

    def sample_links(self, drone_positions, target_pos, obstacles, k_samples=20, return_dbm=False):
        """드론 수와 상관없이 RSSI 샘플 생성"""
        drone_positions = np.asarray(drone_positions, dtype=np.float32)
        target_pos = np.asarray(target_pos, dtype=np.float32)

        n = len(drone_positions)
        out = np.zeros((n, k_samples), dtype=np.float32)
        for i in range(n):
            out[i] = self.sample_one_link(
                tx_pos=target_pos,
                rx_pos=drone_positions[i],
                obstacles=obstacles,
                k_samples=k_samples,
                return_dbm=return_dbm,
            )
        return out

    def sample_three_links(self, drone_positions, target_pos, obstacles, k_samples=20, return_dbm=False):
        """기존 코드 호환용: 드론 3대 RSSI 샘플 생성"""
        return self.sample_links(
            drone_positions=drone_positions,
            target_pos=target_pos,
            obstacles=obstacles,
            k_samples=k_samples,
            return_dbm=return_dbm,
        )
