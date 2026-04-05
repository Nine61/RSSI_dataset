import heapq
import math
import numpy as np
from env import MAP_SIZE, OBS_RADIUS

# A* 설정값
GRID_STEP = 10.0
SAFE_MARGIN = 12.0      # 장애물 추가 여유 거리
WALL_MARGIN = 12.0      # 벽 추가 여유 거리
SMOOTH_CHECK_STEP = 5.0 # 경로 스무딩 시 안전 검사 간격


def world_to_grid(pos):
    gx = int(round(float(pos[0]) / GRID_STEP))
    gy = int(round(float(pos[1]) / GRID_STEP))
    return gx, gy


def grid_to_world(cell):
    gx, gy = cell
    return np.array([gx * GRID_STEP, gy * GRID_STEP], dtype=np.float32)


def inside_grid(cell, occ):
    gx, gy = cell
    h, w = occ.shape
    return 0 <= gx < w and 0 <= gy < h


def build_occupancy_grid(env):
    # 장애물과 벽에 여유를 둔 occupancy grid 생성
    grid_w = int(MAP_SIZE // GRID_STEP) + 1
    grid_h = int(MAP_SIZE // GRID_STEP) + 1
    occ = np.zeros((grid_h, grid_w), dtype=np.uint8)

    inflated_r = OBS_RADIUS + SAFE_MARGIN

    for gy in range(grid_h):
        for gx in range(grid_w):
            x = gx * GRID_STEP
            y = gy * GRID_STEP

            # 벽 근처는 막음
            if x < WALL_MARGIN or x > MAP_SIZE - WALL_MARGIN or y < WALL_MARGIN or y > MAP_SIZE - WALL_MARGIN:
                occ[gy, gx] = 1
                continue

            p = np.array([x, y], dtype=np.float32)
            for obs in env.obstacles:
                if np.linalg.norm(p - np.asarray(obs, dtype=np.float32)) <= inflated_r:
                    occ[gy, gx] = 1
                    break

    return occ


def nearest_free_cell(cell, occ, max_radius=15):
    # 시작점/목표점이 막힌 칸이면 주변 빈 칸으로 보정
    if inside_grid(cell, occ):
        gx, gy = cell
        if occ[gy, gx] == 0:
            return cell

    cx, cy = cell
    for r in range(1, max_radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                cand = (cx + dx, cy + dy)
                if inside_grid(cand, occ):
                    gx, gy = cand
                    if occ[gy, gx] == 0:
                        return cand
    return None


def build_clearance_map(occ):
    # 장애물/벽 근처 칸에 추가 cost 부여
    h, w = occ.shape
    clearance = np.zeros((h, w), dtype=np.float32)

    for gy in range(h):
        for gx in range(w):
            if occ[gy, gx] == 1:
                continue

            penalty = 0.0
            for r in range(1, 4):
                found_block = False
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < w and 0 <= ny < h and occ[ny, nx] == 1:
                            found_block = True
                            break
                    if found_block:
                        break
                if found_block:
                    penalty += (4 - r) * 2.0

            clearance[gy, gx] = penalty

    return clearance


def astar_cells(start_cell, goal_cell, occ, clearance_map=None):
    # 여유 공간을 선호하는 A*
    start_cell = nearest_free_cell(start_cell, occ)
    goal_cell = nearest_free_cell(goal_cell, occ)

    if start_cell is None or goal_cell is None:
        return None

    if clearance_map is None:
        clearance_map = build_clearance_map(occ)

    neighbors = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1)
    ]

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_heap = []
    heapq.heappush(open_heap, (0.0, start_cell))
    came_from = {}
    g_score = {start_cell: 0.0}
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)

        if current == goal_cell:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if not inside_grid(nxt, occ):
                continue

            gx, gy = nxt
            if occ[gy, gx] == 1:
                continue

            # 대각선 코너 끼기 방지
            if dx != 0 and dy != 0:
                side1 = (current[0] + dx, current[1])
                side2 = (current[0], current[1] + dy)
                if inside_grid(side1, occ) and inside_grid(side2, occ):
                    if occ[side1[1], side1[0]] == 1 or occ[side2[1], side2[0]] == 1:
                        continue

            move_cost = math.hypot(dx, dy)
            tentative_g = g_score[current] + move_cost + float(clearance_map[gy, gx])

            if nxt not in g_score or tentative_g < g_score[nxt]:
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f = tentative_g + heuristic(nxt, goal_cell)
                heapq.heappush(open_heap, (f, nxt))

    return None


def segment_is_safe(a, b, env):
    # 두 점을 직선으로 이었을 때 장애물/벽과 충분히 떨어져 있는지 검사
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    total_len = float(np.linalg.norm(b - a))
    if total_len < 1e-9:
        return True

    num = max(2, int(math.ceil(total_len / SMOOTH_CHECK_STEP)))
    ts = np.linspace(0.0, 1.0, num)

    for t in ts:
        p = (1.0 - t) * a + t * b

        if p[0] < WALL_MARGIN or p[0] > MAP_SIZE - WALL_MARGIN:
            return False
        if p[1] < WALL_MARGIN or p[1] > MAP_SIZE - WALL_MARGIN:
            return False

        for obs in env.obstacles:
            if np.linalg.norm(p - np.asarray(obs, dtype=np.float32)) <= (OBS_RADIUS + SAFE_MARGIN):
                return False

    return True


def smooth_path_points(points, env):
    # A* 경로의 지그재그를 줄이기 위해 가능한 구간은 직선으로 연결
    if points is None or len(points) <= 2:
        return points

    smoothed = [points[0]]
    i = 0

    while i < len(points) - 1:
        j = len(points) - 1
        found = False

        while j > i + 1:
            if segment_is_safe(points[i], points[j], env):
                smoothed.append(points[j])
                i = j
                found = True
                break
            j -= 1

        if not found:
            smoothed.append(points[i + 1])
            i += 1

    return np.array(smoothed, dtype=np.float32)


def cells_to_points(cells):
    if cells is None or len(cells) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([grid_to_world(c) for c in cells], dtype=np.float32)


def points_to_dense_trajectory(points, step=10.0):
    # 스무딩된 경유점을 실제 이동용 촘촘한 trajectory로 변환
    points = np.asarray(points, dtype=np.float32)

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(points) == 1:
        return points.copy()

    traj = [points[0]]
    for i in range(len(points) - 1):
        a = points[i]
        b = points[i + 1]
        vec = b - a
        dist = float(np.linalg.norm(vec))
        if dist < 1e-9:
            continue

        direction = vec / dist
        n_steps = max(1, int(math.ceil(dist / step)))
        for k in range(1, n_steps + 1):
            d = min(k * step, dist)
            traj.append(a + direction * d)

    return np.array(traj, dtype=np.float32)


def pad_to_length(traj, target_len):
    traj = np.asarray(traj, dtype=np.float32)

    if len(traj) >= target_len:
        return traj
    if len(traj) == 0:
        raise ValueError("빈 trajectory는 pad_to_length 할 수 없습니다.")

    last = traj[-1]
    hold = np.repeat(last[None, :], target_len - len(traj), axis=0)
    return np.vstack([traj, hold])


def safe_path_trajectory(env, start_pos, goal_pos):
    # 시작점 -> 목표점 안전 경로 전체 생성
    occ = build_occupancy_grid(env)
    clearance_map = build_clearance_map(occ)

    start_cell = world_to_grid(start_pos)
    goal_cell = world_to_grid(goal_pos)

    cells = astar_cells(start_cell, goal_cell, occ, clearance_map)
    if cells is None:
        return np.array([np.asarray(start_pos, dtype=np.float32)], dtype=np.float32)

    coarse_points = cells_to_points(cells)
    smooth_points = smooth_path_points(coarse_points, env)
    traj = points_to_dense_trajectory(smooth_points, step=10.0)

    # 시작점이 중복되면 제거
    if len(traj) > 0 and np.linalg.norm(traj[0] - np.asarray(start_pos, dtype=np.float32)) < 1e-6:
        traj = traj[1:]

    if len(traj) == 0:
        traj = np.array([np.asarray(start_pos, dtype=np.float32)], dtype=np.float32)

    return traj


def choose_assignment_by_path_length(env, start_a, start_b, cand_1, cand_2):
    # 두 follower를 두 후보 위치에 배치할 때 총 이동량이 더 작은 조합 선택
    path_a1 = safe_path_trajectory(env, start_a, cand_1)
    path_b2 = safe_path_trajectory(env, start_b, cand_2)
    cost_1 = len(path_a1) + len(path_b2)

    path_a2 = safe_path_trajectory(env, start_a, cand_2)
    path_b1 = safe_path_trajectory(env, start_b, cand_1)
    cost_2 = len(path_a2) + len(path_b1)

    if cost_1 <= cost_2:
        return (cand_1, path_a1), (cand_2, path_b2)
    else:
        return (cand_2, path_a2), (cand_1, path_b1)
