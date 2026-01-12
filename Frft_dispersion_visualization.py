import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from math import pi

# 1. LOADING DISPERSION DATASET
data = np.load("period_phase_group.npz", allow_pickle=True)
key = data.files[0]
arr = data[key]

if arr.ndim == 3 and arr.shape[2] == 3:
    period = arr[:, :, 0]
    phase  = arr[:, :, 1]
    group  = arr[:, :, 2]
elif arr.ndim == 3 and arr.shape[1] == 3:
    period = arr[:, 0, :]
    phase  = arr[:, 1, :]
    group  = arr[:, 2, :]
else:
    raise ValueError("Unknown data format")

print("Data loaded:", period.shape)


# 2. DISCRETE FFT
def frft_1d(x, a):
    """
    Fractional Fourier Transform
    a = 0   -> identity
    a = 1   -> FFT
    """
    N = len(x)

    if np.isclose(a, 0):
        return x.copy()

    if np.isclose(a, 1):
        return fft(x)

    alpha = a * pi / 2
    tana2 = np.tan(alpha / 2)

    n = np.arange(N)
    chirp = np.exp(-1j * np.pi * n**2 * tana2 / N)

    x1 = x * chirp
    X1 = fft(x1)
    X2 = X1 * chirp

    return ifft(X2)


# 3. FrFT APPLIED
def frft_dispersion_curves(phase, group, a):
    phase_frft = frft_1d(phase, a)
    group_frft = frft_1d(group, a)
    return phase_frft, group_frft

# 4. PLOT OF DISPERSION CURVES
angles = {
    r"$\theta = 0$"     : 0.0,
    r"$\theta = \pi/6$" : 1/3,
    r"$\theta = \pi/4$" : 0.5,
    r"$\theta = \pi/3$" : 2/3,
    r"$\theta = \pi/2$" : 1.0
}

sample_id = 0  # representative sample

for label, a in angles.items():

    phase_t, group_t = frft_dispersion_curves(
        phase[sample_id],
        group[sample_id],
        a
    )

    plt.figure(figsize=(6, 4))

    if np.isclose(a, 0): 
        plt.plot(period[sample_id], np.real(phase_t), "--", linewidth=2, label="Phase velocity")
        plt.plot(period[sample_id], np.real(group_t), "-", linewidth=2, label="Group velocity")
        plt.xlabel("Periods (s)")
        plt.ylabel("S-wave Velocity (km/s)")
    else:  
        frac_domain = np.arange(len(phase_t))
        plt.plot(frac_domain, np.abs(phase_t), "--", linewidth=2, label="Phase amplitude")
        plt.plot(frac_domain, np.abs(group_t), "-", linewidth=2, label="Group amplitude")
        plt.xlabel("Fractional domain")
        plt.ylabel("Amplitude")

    plt.title(f"FrFT Dispersion Curves ({label})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # DISPLAY PLOT
    plt.show()

