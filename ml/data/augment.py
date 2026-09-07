import numpy as np

def mirror_hand(sequence: np.ndarray) -> np.ndarray:
    mirrored = sequence.copy()
    mirrored[..., 0] *= -1
    return mirrored

def add_gaussian_noise(sequence: np.ndarray, std: float = 0.01) -> np.ndarray:
    noise = np.random.normal(0, std, sequence.shape).astype(np.float32)
    return sequence + noise

def time_jitter(sequence: np.ndarray, max_shift: int = 2) -> np.ndarray:
    T = sequence.shape[0]
    idx = np.clip(
        np.arange(T) + np.random.randint(-max_shift, max_shift + 1, size=T),
        0, T - 1,
    )
    return sequence[idx]