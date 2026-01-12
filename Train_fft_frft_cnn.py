import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from math import pi
from scipy.fftpack import fft, ifft

# 1. LOAD DATASET
disp_npz = np.load("period_phase_group.npz")
depth_vs = np.load("depth_vs.npz")

X = disp_npz['data']      
y = depth_vs['data']      

print("Raw shapes -> X:", X.shape, "y:", y.shape)


# 2. PREPROCESSING
ns = X.shape[0]

X_flat = X.reshape(ns, -1)
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X_flat).astype("float32")
X_scaled = X_scaled.reshape(ns, X.shape[1], X.shape[2])

y_flat = y.reshape(ns, -1)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y_flat).astype("float32")

# Train-test split
n_train = int(0.8 * ns)
trainX, testX = X_scaled[:n_train], X_scaled[n_train:]
trainY, testY = y_scaled[:n_train], y_scaled[n_train:]

train_ds = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY))
test_ds = TensorDataset(torch.from_numpy(testX), torch.from_numpy(testY))

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=16)

# 3. FRACTIONAL FOURIER TRANSFORM (FrFT)
def frft(x, a):
    """1D Fractional Fourier Transform."""
    N = len(x)

    if np.isclose(a, 0):
        return x  # identity

    if np.isclose(a, 1):
        return np.real(fft(x))  # FFT

    alpha = a * pi / 2
    tana2 = np.tan(alpha / 2)

    n = np.arange(N)
    chirp1 = np.exp(-1j * np.pi * n**2 * tana2 / N)
    x_mod = x * chirp1

    Xf = fft(x_mod)

    chirp2 = np.exp(-1j * np.pi * n**2 * tana2 / N)
    Xf = Xf * chirp2

    return np.real(ifft(Xf))

def apply_frft_batch(X_batch, a):
    out = np.zeros_like(X_batch)
    for i in range(X_batch.shape[0]):
        for c in range(X_batch.shape[1]):
            out[i, c] = frft(X_batch[i, c], a)
    return out.astype("float32")


# 4. STANDARD FFT
def apply_fft_batch(X_batch):
    out = np.zeros_like(X_batch)
    for i in range(X_batch.shape[0]):
        for c in range(X_batch.shape[1]):
            out[i, c] = np.real(fft(X_batch[i, c]))
    return out.astype("float32")

# 5. CNN MODEL

class DispersionCNN(nn.Module):
    def __init__(self, output_size, nchannels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(nchannels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 6. TRAINING FUNCTION
def train_model(transform_fn, label):
    print(f"\nTraining model: {label}")
    
    model = DispersionCNN(output_size=trainY.shape[1], nchannels=trainX.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses, val_losses = [], []

    for ep in range(10):
        model.train()
        total_train = 0

        for xb, yb in train_dl:
            xb_np = transform_fn(xb.numpy())
            xb = torch.from_numpy(xb_np).to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        train_losses.append(total_train / len(train_dl))

        # Validation
        model.eval()
        total_val = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb_np = transform_fn(xb.numpy())
                xb = torch.from_numpy(xb_np).to(device)
                yb = yb.to(device)
                total_val += criterion(model(xb), yb).item()

        val_losses.append(total_val / len(test_dl))

        print(f"Epoch {ep+1}: Train={train_losses[-1]:.5f}, Val={val_losses[-1]:.5f}")

    return train_losses, val_losses

from math import pi

frft_orders = [
    0.0,
    pi/9,
    pi/8,
    pi/7,
    pi/6,
    pi/5,
    pi/4,
    pi/3
]

frft_labels = [
    "0.00",
    "pi/9",
    "pi/8",
    "pi/7",
    "pi/6",
    "pi/5",
    "pi/4",
    "pi/3"
]


# 7. RUN EXPERIMENT
results = {}

# FFT
results["FFT"] = train_model(apply_fft_batch, "FFT")

# FrFT experiments
for a, lbl in zip(frft_orders, frft_labels):
    label = f"FrFT a={lbl}"
    results[label] = train_model(
        lambda x, aa=a: apply_frft_batch(x, aa),
        label
    )


# 8. FINAL PLOT — ONLY FFT + SELECTED FrFT ORDERS
plt.figure(figsize=(12,6))

plot_keys = [
    "FFT",
    "FrFT a=0.00",
    "FrFT a=pi/9",
    "FrFT a=pi/8",
    "FrFT a=pi/7",
    "FrFT a=pi/6",
    "FrFT a=pi/5",
    "FrFT a=pi/4",
    "FrFT a=pi/3"
]


for key in plot_keys:
    if key in results:
        _, val_losses = results[key]
        plt.plot(val_losses, label=key, linewidth=2)
    else:
        print(f"Warning: {key} missing from results!")

plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("validation_mse_fft_frft.png", dpi=300, bbox_inches="tight")
plt.show()


# 9. OPTIMAL a (MINIMUM VALIDATION MSE)
best_label = None
best_mse = float("inf")
best_epoch = None

for key, (train_losses, val_losses) in results.items():
    min_val = min(val_losses)
    min_epoch = val_losses.index(min_val) + 1

    if min_val < best_mse:
        best_mse = min_val
        best_label = key
        best_epoch = min_epoch

print("\n================ OPTIMAL RESULT ================")
print(f"Best Transform : {best_label}")
print(f"Minimum MSE    : {best_mse:.6f}")
print(f"At Epoch       : {best_epoch}")
print("================================================")

