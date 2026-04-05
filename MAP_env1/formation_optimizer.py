import math
import numpy as np
from env import MAP_SIZE, OBS_RADIUS
from rssi_env import RealisticRSSIEnv


def distance_point_to_segment(point, seg_a, seg_b):
    """점과 선분 사이의 최단거리 계산"""
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


def make_equilateral_vertices(center, radius, theta_deg):
    """center를 중심으로 반경 radius인 정삼각형 꼭짓점 3개 생성"""
    center = np.asarray(center, dtype=np.float32)
    verts = []
    for k in range(3):
        ang = np.deg2rad(theta_deg + 120.0 * k)
        verts.append([
            center[0] + radius * math.cos(ang),
            center[1] + radius * math.sin(ang)
        ])
    return np.asarray(verts, dtype=np.float32)


def in_bounds(points, wall_margin=20.0):
    """꼭짓점이 벽 여유 범위 안에 모두 들어오는지 확인"""
    points = np.asarray(points, dtype=np.float32)
    xs = points[:, 0]
    ys = points[:, 1]
    return bool(
        np.all(xs >= wall_margin) and
        np.all(xs <= MAP_SIZE - wall_margin) and
        np.all(ys >= wall_margin) and
        np.all(ys <= MAP_SIZE - wall_margin)
    )


def wall_penalty(points, wall_margin=20.0):
    """벽에 가까울수록 페널티 증가"""
    points = np.asarray(points, dtype=np.float32)
    penalty = 0.0
    for p in points:
        x, y = float(p[0]), float(p[1])
        if x < wall_margin:
            penalty += (wall_margin - x)
        if x > MAP_SIZE - wall_margin:
            penalty += (x - (MAP_SIZE - wall_margin))
        if y < wall_margin:
            penalty += (wall_margin - y)
        if y > MAP_SIZE - wall_margin:
            penalty += (y - (MAP_SIZE - wall_margin))
    return float(penalty)


def obstacle_penalty(points, obstacles, obstacle_margin=20.0):
    """꼭짓점이 장애물에 가깝거나 내부에 있으면 페널티"""
    points = np.asarray(points, dtype=np.float32)
    penalty = 0.0
    safe_r = OBS_RADIUS + obstacle_margin

    for p in points:
        for obs in obstacles:
            obs = np.asarray(obs, dtype=np.float32)
            d = float(np.linalg.norm(p - obs))
            if d < safe_r:
                penalty += (safe_r - d)

    return float(penalty)


def link_obstacle_penalty(points, target_pos, obstacles, near_margin=20.0):
    """타겟-드론 링크가 장애물을 관통하거나 가까이 지나가면 페널티"""
    points = np.asarray(points, dtype=np.float32)
    target_pos = np.asarray(target_pos, dtype=np.float32)
    penalty = 0.0

    for p in points:
        for obs in obstacles:
            obs = np.asarray(obs, dtype=np.float32)
            d = distance_point_to_segment(obs, target_pos, p)

            if d <= OBS_RADIUS:
                penalty += 5.0
            elif d <= OBS_RADIUS + near_margin:
                penalty += 2.0

    return float(penalty)


def geometry_score(points, target_pos):
    """타겟이 중심에 가깝고 드론 간 간격이 균형적일수록 높은 점수"""
    points = np.asarray(points, dtype=np.float32)
    target_pos = np.asarray(target_pos, dtype=np.float32)

    centroid = np.mean(points, axis=0)
    center_err = float(np.linalg.norm(centroid - target_pos))

    d01 = float(np.linalg.norm(points[0] - points[1]))
    d12 = float(np.linalg.norm(points[1] - points[2]))
    d20 = float(np.linalg.norm(points[2] - points[0]))
    edge_std = float(np.std([d01, d12, d20]))

    return float(100.0 - (2.0 * center_err + 1.5 * edge_std))


def rssi_quality_score(points, target_pos, obstacles, rssi_env=None, k_samples=20):
    """평균 RSSI는 높고, 표준편차는 작은 배치를 선호"""
    points = np.asarray(points, dtype=np.float32)
    target_pos = np.asarray(target_pos, dtype=np.float32)

    if rssi_env is None:
        rssi_env = RealisticRSSIEnv(seed=123)

    samples = rssi_env.sample_three_links(
        drone_positions=points,
        target_pos=target_pos,
        obstacles=obstacles,
        k_samples=k_samples,
    )

    means = np.mean(samples, axis=1)
    stds = np.std(samples, axis=1)
    return float(np.mean(means) * 100.0 - np.mean(stds) * 100.0)


def evaluate_candidate(points, target_pos, obstacles, rssi_env=None):
    """후보 배치 1개를 종합 평가"""
    if rssi_env is None:
        rssi_env = RealisticRSSIEnv(seed=123)

    if not in_bounds(points):
        return {
            "total_score": -1e9,
            "geometry_score": -1e9,
            "rssi_score": -1e9,
            "wall_penalty": 1e9,
            "obstacle_penalty": 1e9,
            "link_penalty": 1e9,
        }

    g_score = geometry_score(points, target_pos)
    r_score = rssi_quality_score(points, target_pos, obstacles, rssi_env=rssi_env)
    w_pen = wall_penalty(points)
    o_pen = obstacle_penalty(points, obstacles)
    l_pen = link_obstacle_penalty(points, target_pos, obstacles)

    total = (
        1.0 * g_score +
        1.2 * r_score -
        4.0 * w_pen -
        4.0 * o_pen -
        6.0 * l_pen
    )

    return {
        "total_score": float(total),
        "geometry_score": float(g_score),
        "rssi_score": float(r_score),
        "wall_penalty": float(w_pen),
        "obstacle_penalty": float(o_pen),
        "link_penalty": float(l_pen),
    }


def generate_candidate_formations(target_pos, radius_list=None, angle_step_deg=15.0):
    """타겟 중심 기준으로 여러 정삼각형 후보 생성"""
    if radius_list is None:
        radius_list = [45.0, 55.0, 65.0, 75.0, 85.0, 95.0]

    candidates = []
    for radius in radius_list:
        theta = 0.0
        while theta < 360.0:
            points = make_equilateral_vertices(
                center=target_pos,
                radius=radius,
                theta_deg=theta,
            )
            candidates.append({
                "radius": float(radius),
                "theta_deg": float(theta),
                "points": points,
            })
            theta += angle_step_deg
    return candidates


def find_best_formation(target_pos, obstacles, radius_list=None, angle_step_deg=15.0, rssi_env=None):
    """후보 정삼각형들을 평가해서 최적 배치 반환"""
    if rssi_env is None:
        rssi_env = RealisticRSSIEnv(seed=123)

    candidates = generate_candidate_formations(
        target_pos=target_pos,
        radius_list=radius_list,
        angle_step_deg=angle_step_deg,
    )

    best = None
    best_eval = None

    for cand in candidates:
        ev = evaluate_candidate(
            points=cand["points"],
            target_pos=target_pos,
            obstacles=obstacles,
            rssi_env=rssi_env,
        )

        if best is None or ev["total_score"] > best_eval["total_score"]:
            best = cand
            best_eval = ev

    return {
        "best_points": best["points"],
        "radius": best["radius"],
        "theta_deg": best["theta_deg"],
        "evaluation": best_eval,
    }
