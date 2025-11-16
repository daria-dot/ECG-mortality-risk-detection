import numpy as np
import app

def gen_normal_candidate(seed: int) -> np.ndarray:
    length = 4096
    n_leads = 8
    fs = 400
    duration = length / fs
    t = np.linspace(0, duration, length)

    base_hr_hz = 1.2
    noise_scale = 0.03
    st_deviation = 0.0

    rng = np.random.default_rng(seed)
    ecg = np.zeros((length, n_leads), dtype=np.float32)

    for lead in range(n_leads):
        phase = rng.uniform(0, 2 * np.pi)
        signal = (
            0.6 * np.sin(2 * np.pi * base_hr_hz * t + phase) +
            0.2 * np.sin(2 * np.pi * base_hr_hz * 2 * t + phase / 2) +
            0.1 * np.sin(2 * np.pi * base_hr_hz * 3 * t - phase / 3)
        )
        signal = signal + st_deviation
        scale = 1.0 + 0.1 * (lead - n_leads / 2) / (n_leads / 2)
        noise = noise_scale * rng.standard_normal(length)
        ecg[:, lead] = (signal * scale + noise).astype(np.float32)
    return ecg


def main():
    model = app.load_model()

    best_seed = None
    best_s6 = -1.0

    for seed in range(1, 51):
        ecg = gen_normal_candidate(seed)
        X = app.process_ecg_input(ecg)
        _, cum = app.predict_survival(model, X)
        s6 = float(cum[6])
        print(f"seed={seed} S6={s6}")
        if s6 > best_s6:
            best_s6 = s6
            best_seed = seed

    print("BEST_SEED", best_seed, "BEST_S6", best_s6)


if __name__ == "__main__":
    main()
