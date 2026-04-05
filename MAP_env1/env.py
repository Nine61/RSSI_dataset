import numpy as np

# =============================================
# --- 환경 설정 상수 (test.py와 train.py가 참조함) ---
# =============================================
MAP_SIZE              = 700
START_POS             = np.array([30.0, 30.0])

OBS_COUNT             = 5
OBS_RADIUS            = 40

BPSK_SIGNAL_RADIUS    = 250
QAM_SIGNAL_RADIUS     = 150

TARGET_RSSI_THRESHOLD = 0.85
MAX_STEPS             = 500

TARGET_MODE           = 'random'

OBS_SOFT_MARGIN       = 100
OBS_HARD_MARGIN       = 55

# 💡 [주의] 절대 'from env import ...' 코드를 여기에 넣지 마세요!

class DroneEnv:
    """
    700×700 2D 드론 환경 (팀원들의 29차원 설계 반영)
    """
    def __init__(self):
        self.map_size    = MAP_SIZE
        self.drone_start = START_POS.copy()
        self.state_dim   = 29
        self.action_dim  = 2

        self.obstacles = [
            np.array([200.0, 200.0]),
            np.array([450.0, 150.0]),
            np.array([150.0, 480.0]),
            np.array([500.0, 450.0]),
            np.array([330.0, 350.0]),
        ]

    def reset(self):
        self.drone_pos   = self.drone_start.copy()
        self.prev_action = np.array([1.0, 1.0]) / np.sqrt(2)
        self.search_vec  = np.array([1.0, 1.0]) / np.sqrt(2)
        self.steps       = 0

        self.prev_bpsk_rssi = 0.0
        self.prev_qam0_rssi = 0.0
        self.prev_qam1_rssi = 0.0

        self.GRID_N      = 14
        self.visit_count = np.zeros((self.GRID_N, self.GRID_N), dtype=np.int32)

        if TARGET_MODE == 'manual':
            # 매뉴얼 모드 변수들 (상단 정의 필요 시 추가)
            self.bpsk_pos = np.array([500.0, 550.0]) 
            self.qam_pos  = [np.array([200.0, 500.0]), np.array([550.0, 200.0])]
        else:
            self.bpsk_pos = self._spawn_signal(min_gap_obs=OBS_RADIUS + 30, min_gap_start=220, others=[])
            qam0 = self._spawn_signal(min_gap_obs=OBS_RADIUS + 20, min_gap_start=150, others=[self.bpsk_pos], min_gap_others=200)
            qam1 = self._spawn_signal(min_gap_obs=OBS_RADIUS + 20, min_gap_start=150, others=[self.bpsk_pos, qam0], min_gap_others=200)
            self.qam_pos = [qam0, qam1]

        return self.get_state()

    def _spawn_signal(self, min_gap_obs, min_gap_start, others, min_gap_others=0):
        while True:
            pos = np.random.uniform(50, self.map_size - 50, size=2)
            if pos[0] <= min_gap_start and pos[1] <= min_gap_start: continue
            if any(np.linalg.norm(pos - obs) < min_gap_obs for obs in self.obstacles): continue
            if others and any(np.linalg.norm(pos - o) < min_gap_others for o in others): continue
            return pos

    def get_bpsk_rssi(self, pos):
        dist = np.linalg.norm(self.bpsk_pos - pos)
        if dist < BPSK_SIGNAL_RADIUS:
            return 1.0 - dist / BPSK_SIGNAL_RADIUS
        return 0.0

    def get_qam_rssi(self, pos, idx):
        dist = np.linalg.norm(self.qam_pos[idx] - pos)
        if dist < QAM_SIGNAL_RADIUS:
            return 1.0 - dist / QAM_SIGNAL_RADIUS
        return 0.0

    def get_state(self):
        curr_bpsk = self.get_bpsk_rssi(self.drone_pos)
        curr_qam0 = self.get_qam_rssi(self.drone_pos, 0)
        curr_qam1 = self.get_qam_rssi(self.drone_pos, 1)

        bpsk_diff = curr_bpsk - self.prev_bpsk_rssi
        qam0_diff = curr_qam0 - self.prev_qam0_rssi
        qam1_diff = curr_qam1 - self.prev_qam1_rssi

        if curr_bpsk > 0.0:
            vec = self.bpsk_pos - self.drone_pos
            n = np.linalg.norm(vec)
            sensor_dir = vec / n if n > 0.001 else np.zeros(2)
        else:
            sensor_dir = self.search_vec

        state = [
            self.drone_pos[0] / self.map_size, self.drone_pos[1] / self.map_size,
            (self.map_size - self.drone_pos[0]) / self.map_size, (self.map_size - self.drone_pos[1]) / self.map_size,
            curr_bpsk, bpsk_diff * 100.0,
            self.prev_action[0], self.prev_action[1],
            sensor_dir[0], sensor_dir[1],
            curr_qam0, qam0_diff * 100.0,
            curr_qam1, qam1_diff * 100.0,
        ]

        for obs in self.obstacles:
            rx = (obs[0] - self.drone_pos[0]) / self.map_size
            ry = (obs[1] - self.drone_pos[1]) / self.map_size
            dist = np.linalg.norm([rx, ry])
            state.extend([rx, ry, dist])

        return np.array(state, dtype=np.float32)

    def _apf_repulsion(self, pos, velocity):
        APF_INFLUENCE, APF_K_REP, APF_MAX = 100.0, 90000.0, 0.75
        correction = np.zeros(2)
        for obs in self.obstacles:
            d = np.linalg.norm(pos - obs)
            if d >= APF_INFLUENCE or d < 1.0: continue
            rep_dir = (pos - obs) / d
            magnitude = APF_K_REP * (1.0/d - 1.0/APF_INFLUENCE) / (d * d)
            normal_proj = np.dot(velocity, rep_dir)
            if normal_proj < 0:
                tangent_vec = velocity - rep_dir * normal_proj
                t_n = np.linalg.norm(tangent_vec)
                tangent_boost = (tangent_vec / t_n * magnitude * 2.0) if t_n > 0.001 else (np.array([-rep_dir[1], rep_dir[0]]) * magnitude * 2.0)
                correction += rep_dir * magnitude + tangent_boost
            else: correction += rep_dir * magnitude * 0.5
        
        # 벽 척력
        W_INF, W_K = 80.0, 60000.0
        wall_reps = [(pos[0], np.array([1,0])), (700-pos[0], np.array([-1,0])), (pos[1], np.array([0,1])), (700-pos[1], np.array([0,-1]))]
        for d_w, r_d in wall_reps:
            if d_w < W_INF and d_w >= 1.0:
                mag = W_K * (1.0/d_w - 1.0/W_INF) / (d_w * d_w)
                correction += r_d * mag * (2.0 if np.dot(velocity, r_d) < 0 else 0.3)

        c_n = np.linalg.norm(correction)
        return (correction / c_n * APF_MAX) if c_n > APF_MAX else correction

    def _obs_penalty(self, pos):
        penalty = 0.0
        for obs in self.obstacles:
            d = np.linalg.norm(pos - obs)
            if OBS_HARD_MARGIN < d < OBS_SOFT_MARGIN:
                penalty -= ((OBS_SOFT_MARGIN - d) / (OBS_SOFT_MARGIN - OBS_HARD_MARGIN)) * 200.0
            elif OBS_RADIUS < d <= OBS_HARD_MARGIN:
                penalty -= ((OBS_HARD_MARGIN - d) / (OBS_HARD_MARGIN - OBS_RADIUS))**2 * 800.0
        return penalty

    def step(self, action):
        move_step = 10.0
        action = np.clip(action, -1.0, 1.0)
        a_n = np.linalg.norm(action)
        action = action / a_n if a_n > 0.001 else self.prev_action

        inertia = 0.5
        velocity = action * (1.0 - inertia) + self.prev_action * inertia
        v_n = np.linalg.norm(velocity)
        velocity = velocity / v_n if v_n > 0.001 else self.prev_action

        apf_rep = self._apf_repulsion(self.drone_pos, velocity)
        velocity = velocity + apf_rep
        v_n2 = np.linalg.norm(velocity)
        if v_n2 > 0.001: velocity /= v_n2

        self.prev_action = velocity
        new_pos = self.drone_pos + velocity * move_step

        # 벽 반사 로직
        margin = 70.0
        for i in range(2):
            if (new_pos[i] < margin and self.search_vec[i] < 0) or (new_pos[i] > self.map_size - margin and self.search_vec[i] > 0):
                self.search_vec[i] *= -1
        self.search_vec /= np.linalg.norm(self.search_vec)

        curr_bpsk = self.get_bpsk_rssi(new_pos)
        bpsk_diff = curr_bpsk - self.prev_bpsk_rssi

        reward, done, info = 0.0, False, ""
        
        # 충돌 체크
        if (new_pos[0] < 0 or new_pos[0] > self.map_size or new_pos[1] < 0 or new_pos[1] > self.map_size):
            reward, done, info = -2000.0, True, "Wall Crash"
        for obs in self.obstacles:
            if np.linalg.norm(new_pos - obs) <= OBS_RADIUS:
                reward, done, info = -2000.0, True, "Obstacle Crash"
        
        if not done:
            self.drone_pos = new_pos
            self.steps += 1
            if curr_bpsk >= TARGET_RSSI_THRESHOLD:
                reward, done, info = 3000.0, True, "Goal Reached"
            else:
                # 보상 로직 (추적/탐색 모드)
                if curr_bpsk > 0.0:
                    vec = self.bpsk_pos - self.drone_pos
                    alignment = np.dot(velocity, vec / np.linalg.norm(vec))
                    reward = -1.0 + alignment * 8.0 + bpsk_diff * 300.0 + self._obs_penalty(new_pos)
                else:
                    gx, gy = int(np.clip(new_pos[0]/700*14, 0, 13)), int(np.clip(new_pos[1]/700*14, 0, 13))
                    visit_p = -self.visit_count[gx, gy] * 1.5
                    self.visit_count[gx, gy] += 1
                    reward = -0.5 + np.dot(velocity, self.search_vec) * 10.0 + visit_p + self._obs_penalty(new_pos)

        if self.steps >= MAX_STEPS: done, info = True, "Timeout"
        self._update_prev(curr_bpsk)
        return self.get_state(), reward, done, info

    def _update_prev(self, curr_bpsk):
        self.prev_bpsk_rssi = curr_bpsk
        self.prev_qam0_rssi = self.get_qam_rssi(self.drone_pos, 0)
        self.prev_qam1_rssi = self.get_qam_rssi(self.drone_pos, 1)