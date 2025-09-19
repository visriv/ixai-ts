import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_locality(curve, out, label="Locality", fits=None, mode="mean"):
    """
    Plot locality curve(s).
    Args:
      curve: np.ndarray[tau_max+1] or [tau_max+1, N]
      out: path to save
      label: str
      fits: dict with 'exp' or 'pow' params (applied to aggregate curve)
      mode: 'mean' | 'median' | 'all'
    """
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)

    if curve.ndim == 2:  # [tau_max+1, N]
        taus = np.arange(curve.shape[0])
        if mode == "all":
            plt.figure()
            # plot all curves faintly
            for n in range(curve.shape[1]):
                plt.plot(taus, curve[:, n], alpha=0.2, lw=1, color="gray")
            # aggregate mean and std
            agg = curve.mean(axis=1)
            std = curve.std(axis=1)
            plt.plot(taus, agg, lw=2, color="blue", label=f"{label} (mean ± std)")
            plt.fill_between(taus, agg-std, agg+std, color="blue", alpha=0.2)
        elif mode == "median":
            agg = np.median(curve, axis=1)
            plt.plot(taus, agg, lw=2, label=f"{label} (median)")
        else:  # mean by default
            agg = curve.mean(axis=1)
            plt.plot(taus, agg, lw=2, label=f"{label} (mean)")
    else:  # [tau_max+1]
        taus = np.arange(len(curve))
        agg = curve
        plt.figure()
        plt.plot(taus, agg, lw=2, label=label)

    # Overlay fits if provided
    if fits is not None:
        if 'exp' in fits and len(fits['exp']) == 2:
            a, b = fits['exp']
            plt.plot(taus, a * np.exp(-b * taus), '--', label=f'exp fit b={b:.3f}')
        if 'pow' in fits and len(fits['pow']) == 2:
            a, p = fits['pow']
            plt.plot(taus, a * np.power(taus + 1, -p), '--', label=f'pow fit p={p:.3f}')

    plt.xlabel('Lag τ')
    plt.ylabel('|Interaction|')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()



def plot_spectrum(mag, out, mode="mean"):
    """
    Plot spectrum magnitude.
    Args:
      mag: np.ndarray[F] or [F, N]
      out: path to save
      mode: 'mean' | 'median' | 'all'
    """
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)

    if mag.ndim == 2:  # [F, N]
        freqs = np.arange(mag.shape[0])
        if mode == "all":
            plt.figure()
            for n in range(mag.shape[1]):
                plt.plot(freqs, mag[:, n], alpha=0.3, lw=1)
            agg = mag.mean(axis=1)
        elif mode == "median":
            agg = np.median(mag, axis=1)
        else:
            agg = mag.mean(axis=1)

        plt.plot(freqs, agg, lw=2, label=f"Spectrum ({mode})")
    else:  # [F]
        freqs = np.arange(len(mag))
        agg = mag
        plt.plot(freqs, agg, lw=2, label="Spectrum")

    plt.xlabel('Frequency bin')
    plt.ylabel('|DFT|')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
