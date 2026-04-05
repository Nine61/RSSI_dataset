import numpy as np


class RSSIKalmanFilter:
    # 1차원 RSSI smoothing용 간단한 칼만필터
    def __init__(self, process_var=5e-4, meas_var=2e-3, init_est=0.5, init_err=1.0):
        self.process_var = float(process_var)
        self.meas_var = float(meas_var)
        self.x = float(init_est)
        self.P = float(init_err)

    def update(self, z):
        z = float(z)

        # predict
        self.P = self.P + self.process_var

        # update
        K = self.P / (self.P + self.meas_var)
        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * self.P
        return self.x


def filter_rssi_sequence(rssi_seq, process_var=5e-4, meas_var=2e-3):
    rssi_seq = np.asarray(rssi_seq, dtype=np.float32)
    if len(rssi_seq) == 0:
        return np.array([], dtype=np.float32)

    # 첫 샘플은 초기값으로 그대로 사용
    kf = RSSIKalmanFilter(
        process_var=process_var,
        meas_var=meas_var,
        init_est=float(rssi_seq[0]),
        init_err=1.0,
    )

    filtered = [float(rssi_seq[0])]
    for z in rssi_seq[1:]:
        filtered.append(kf.update(float(z)))

    return np.asarray(filtered, dtype=np.float32)


def filter_links(rssi_samples, process_var=5e-4, meas_var=2e-3):
    # shape: (n_links, k)
    rssi_samples = np.asarray(rssi_samples, dtype=np.float32)
    out = np.zeros_like(rssi_samples, dtype=np.float32)
    for i in range(rssi_samples.shape[0]):
        out[i] = filter_rssi_sequence(
            rssi_samples[i],
            process_var=process_var,
            meas_var=meas_var,
        )
    return out


def filter_three_links(rssi_samples_3xk, process_var=5e-4, meas_var=2e-3):
    return filter_links(
        rssi_samples_3xk,
        process_var=process_var,
        meas_var=meas_var,
    )


def extract_mean_std_features(filtered_rssi):
    filtered_rssi = np.asarray(filtered_rssi, dtype=np.float32)
    means = np.mean(filtered_rssi, axis=1)
    stds = np.std(filtered_rssi, axis=1)
    return np.concatenate([means, stds], axis=0).astype(np.float32)
