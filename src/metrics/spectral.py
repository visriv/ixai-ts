import numpy as np

def dft_magnitude(curve: np.ndarray) -> np.ndarray:
    F = np.fft.rfft(curve)
    return np.abs(F)

def spectral_bandwidth(mag: np.ndarray, mass: float = 0.95) -> int:
    c = np.cumsum(mag) / (mag.sum() + 1e-12)
    k = np.searchsorted(c, mass)
    return int(k)

def spectral_centroid(mag, sr=1.0):
    """
    mag: [F] or [F, N] magnitude spectrum
    sr: sampling rate (default 1.0, only scales frequencies)
    Returns: centroid scalar (if 1D) or [N] array (if 2D)
    """
    mag = np.array(mag)
    F = mag.shape[0]
    freqs = np.linspace(0, sr/2, F)  # [F]

    if mag.ndim == 1:
        return float((freqs * mag).sum() / (mag.sum() + 1e-12))
    elif mag.ndim == 2:
        # Broadcast freqs [F,1] with mag [F,N]
        num = (freqs[:, None] * mag).sum(axis=0)
        den = mag.sum(axis=0) + 1e-12
        return num / den  # shape [N]
    else:
        raise ValueError(f"Unexpected mag shape: {mag.shape}")



def spectral_flatness(mag: np.ndarray) -> np.ndarray:
    """
    Spectral flatness (a.k.a. Wiener entropy).
    mag: [F] or [F, N] magnitude spectrum(s)
    Returns: float (if 1D) or [N] array (if 2D)
    """
    mag = np.array(mag)
    
    if mag.ndim == 1:
        gm = np.exp(np.mean(np.log(mag + 1e-12)))
        am = np.mean(mag + 1e-12)
        return float(gm / am)
    elif mag.ndim == 2:
        # geometric mean across freq axis
        gm = np.exp(np.mean(np.log(mag + 1e-12), axis=0))  # [N]
        am = np.mean(mag + 1e-12, axis=0)                  # [N]
        return gm / am  # [N]
    else:
        raise ValueError(f"Unexpected mag shape: {mag.shape}")

