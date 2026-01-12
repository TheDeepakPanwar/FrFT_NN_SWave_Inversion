import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from math import pi

# 1. LOADING DATA
disp_npz = np.load("period_phase_group.npz")
depth_vs = np.load("depth_vs.npz")

X = disp_npz['data']
y = depth_vs['data']

print("Raw shapes:", X.shape, y.shape)


# 2. PREPROCESSING
ns = X.shape[0]

X_flat = X.reshape(ns, -1)
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X_flat).astype("float32")
X_scaled = X_scaled.reshape(ns, X.shape[1], X.shape[2])

y_flat = y.reshape(ns, -1)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y_flat).astype("float32")

# Train / test split
n_train = int(0.8 * ns)

trainX, testX = X_scaled[:n_train], X_scaled[n_train:]
trainY, testY = y_scaled[:n_train], y_scaled[n_train:]

train_ds = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY))
test_ds  = TensorDataset(torch.from_numpy(testX),  torch.from_numpy(testY))

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl  = DataLoader(test_ds, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 3. FFT & FrFT DEFINITIONS
def frft(x, a):
    N = len(x)
    if np.isclose(a, 0):
        return x
    if np.isclose(a, 1):
        return np.real(fft(x))

    alpha = a * pi / 2
    tana2 = np.tan(alpha / 2)
    n = np.arange(N)

    chirp = np.exp(-1j * np.pi * n**2 * tana2 / N)
    x1 = x * chirp
    Xf = fft(x1)
    Xf = Xf * chirp

    return np.real(ifft(Xf))


def apply_frft_batch(Xb, a):
    out = np.zeros_like(Xb)
    for i in range(Xb.shape[0]):
        for c in range(Xb.shape[1]):
            out[i, c] = frft(Xb[i, c], a)
    return out.astype("float32")


def apply_fft_batch(Xb):
    out = np.zeros_like(Xb)
    for i in range(Xb.shape[0]):
        for c in range(Xb.shape[1]):
            out[i, c] = np.real(fft(Xb[i, c]))
    return out.astype("float32")


# 4. CNN MODEL
class DispersionCNN(nn.Module):
    def __init__(self, output_size, nchannels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(nchannels, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# 5. TRAIN MODEL
def train_model(transform_fn):
    model = DispersionCNN(trainY.shape[1], trainX.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for ep in range(10):
        model.train()
        for xb, yb in train_dl:
            xb = torch.from_numpy(transform_fn(xb.numpy())).to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


# 6. ADDING NOISE (SNR CONTROL)
def add_noise_snr(X, snr_db):
    if np.isinf(snr_db):
        return X.copy()

    signal_power = np.mean(X**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), X.shape)

    return X + noise


# 7. EVALUATION UNDER NOISE
def evaluate_noise(model, transform_fn, snr_db):
    model.eval()
    Xn = add_noise_snr(testX, snr_db)
    Xn = transform_fn(Xn)

    with torch.no_grad():
        pred = model(torch.from_numpy(Xn).to(device))
        mse = nn.MSELoss()(pred, torch.from_numpy(testY).to(device)).item()

    return mse


# 8. NOISE SENSITIVITY ANALYSIS
snr_levels = [np.inf, 30, 20, 10]

transforms = {
    "FFT": apply_fft_batch,
    "FrFT a=0": lambda x: apply_frft_batch(x, 0.0),
    "FrFT a=pi/6": lambda x: apply_frft_batch(x, pi/6),
    "FrFT a=pi/4": lambda x: apply_frft_batch(x, pi/4),
    "FrFT a=pi/3": lambda x: apply_frft_batch(x, pi/3)
}

results = {}

for name, tf in transforms.items():
    print(f"\nTraining {name}")
    model = train_model(tf)

    results[name] = []
    for snr in snr_levels:
        mse = evaluate_noise(model, tf, snr)
        results[name].append(mse)
        print(f"  SNR={snr}: MSE={mse:.5f}")


# 9. PLOT RESULTS
plt.figure(figsize=(8,5))
snr_labels = ["∞", "30", "20", "10"]

for name, vals in results.items():
    plt.plot(snr_labels, vals, marker="o", label=name)

plt.xlabel("SNR (dB)")
plt.ylabel("MSE")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("noise_sensitivity_fft_frft.png", dpi=300)
plt.show()
