import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
from scipy.interpolate import griddata


# === 1. Wczytaj obraz sferyczny ===
image_path = "zdjecia_3d/S1_R4_sferyczne.JPG"
room_image = cv2.imread(image_path)
room_image = cv2.cvtColor(room_image, cv2.COLOR_BGR2RGB)
height, width, _ = room_image.shape


# === 2. Wczytaj dane ambisoniczne ===
wav_path = "pliki_wave/M0001_S01_R04_ambix.wav"
audio_data, sample_rate = sf.read(wav_path)
audio_data = audio_data / np.max(np.abs(audio_data))  # normalizacja
W, X, Y, Z = audio_data[:, 0], audio_data[:, 1], audio_data[:, 2], audio_data[:, 3]

# === 3. Ustawienia ===
frame_duration = 0.002  # 2 ms
samples_per_frame = int(sample_rate * frame_duration)
num_frames = len(W) // samples_per_frame
output_dir = "klatki_animacji"
os.makedirs(output_dir, exist_ok=True)

# === 4. Funkcje pomocnicze ===
def bformat_to_direction(x, y, z):
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x**2 + y**2))
    return np.degrees(az), np.degrees(el)

def direction_to_image_coords(azimuth_deg, elevation_deg, width, height):
    x = (azimuth_deg + 180) / 360 * width
    y = (90 - elevation_deg) / 180 * height
    return int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))

# === 5. Generowanie kolejnych klatek ===
for frame in range(num_frames):
    start = frame * samples_per_frame
    end = start + samples_per_frame
    w = W[start:end]
    x = X[start:end]
    y = Y[start:end]
    z = Z[start:end]

    coords = []
    amps = []

    for i in range(len(w)):
        az, el = bformat_to_direction(x[i], y[i], z[i])
        xi, yi = direction_to_image_coords(az, el, width, height)
        amp = np.abs(w[i])
        coords.append((xi, yi))
        amps.append(amp)

    coords = np.array(coords)
    amps = np.array(amps)
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    heatmap = griddata(coords, amps, (grid_x, grid_y), method='linear', fill_value=0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1  # unormuj

    # === 6. Nałożenie mapy na zdjęcie ===
    plt.figure(figsize=(14, 7))
    plt.imshow(room_image, extent=(0, width, height, 0))
    plt.imshow(heatmap, cmap='hot', alpha=0.6, extent=(0, width, height, 0))
    plt.axis('off')
    plt.colorbar(label='')
    plt.savefig(os.path.join(output_dir, f"frame_{frame:04d}.png"), dpi=150)
    plt.close()
