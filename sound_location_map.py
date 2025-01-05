import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

# 1. Wczytaj obraz panoramiczny
image_path = "obrazek.tiff"
room_image = cv2.imread(image_path)
room_image = cv2.cvtColor(room_image, cv2.COLOR_BGR2RGB)

# 2. Wczytaj plik WAV (odpowiedź impulsowa)
ds_wav_path = "direct_sound.wav"
fr_wav_path = "first_reflection.wav"
sample_rate, audio_data = read(ds_wav_path)

if audio_data.ndim == 1:
    raise ValueError("Plik WAV ma tylko jeden kanał. Oczekiwano 4 kanałów B-formatu.")

W = audio_data[:, 0]

X = audio_data[:, 1]
Y = audio_data[:, 2]
Z = audio_data[:, 3]
print(X)
# # 3. Oblicz amplitudę (maksymalna wartość W)
# amplitudes= []
# for _ in W:
#     _ = np.abs(_)
#     amplitudes.append (_)
#jednak nie chyba
amplitude = np.max(W)
height, width, _ = room_image.shape  # Pobranie wymiarów obrazu
mic_position = (width // 2, height // 2)  # Ustalenie pozycji mikrofonu w centrum obrazu
# 4. Oblicz kierunek (azymut i elewacja)
Ix = W * X
Iy = W * Y
Iz = W * Z

def compute_azimuth_elevation(Ix, Iy, Iz):
    azimuth = np.arctan2(Iy, Ix)
    distance = np.sqrt((Ix - mic_position[0]) ** 2 + (Iy - mic_position[1]) ** 2)  # Odległość od mikrofonu
    elevation = np.arctan2(Iz, distance)

    azimuth_deg = np.degrees(azimuth)
    elevation_deg = np.degrees(elevation)

    return azimuth_deg, elevation_deg, distance


azimuth, elevation, distance = compute_azimuth_elevation(Ix, Iy, Iz)

print(W, X,Y,Z)
print(amplitudes)