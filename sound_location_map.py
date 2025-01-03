import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.interpolate import griddata
import math

# 1. Wczytaj obraz panoramiczny
image_path = "obrazek.tiff"  # Zamień na ścieżkę do obrazu panoramicznego
room_image = cv2.imread(image_path)  # Wczytanie obrazu
room_image = cv2.cvtColor(room_image, cv2.COLOR_BGR2RGB)  # Konwersja z BGR (OpenCV) na RGB (matplotlib)

# 2. Wczytaj plik WAV (odpowiedź impulsowa)
wav_path = "ambisonic_impulse_response.wav"  # Ścieżka do pliku WAV
sample_rate, audio_data = read(wav_path)

if audio_data.ndim == 1:
    raise ValueError("Plik WAV ma tylko jeden kanał. Oczekiwano 4 kanałów B-formatu.")

W = audio_data[:, 0]
X = audio_data[:, 1]
Y = audio_data[:, 2]
Z = audio_data[:, 3]

Ix = W * X
Iy = W * Y
Iz = W * Z

# 3. Ustaw mikrofon w centrum obrazu
height, width, _ = room_image.shape
microphone_position = (width // 2, height // 2)

# 4. Obliczanie azymutu i elewacji
def compute_azimuth_elevation(Ix, Iy, Iz):
    azimuth = np.arctan2(Iy, Ix)
    distance = np.sqrt(np.maximum(Ix ** 2 + Iy ** 2, 0))  # Ograniczenie do nieujemnych wartości
    elevation = np.arctan2(Iz, distance)

    azimuth_deg = np.degrees(azimuth)
    elevation_deg = np.degrees(elevation)

    return azimuth_deg, elevation_deg

azimuth, elevation = compute_azimuth_elevation(Ix, Iy, Iz)

# 5. Definiowanie siatki i promienia
x = azimuth
y = elevation

h = 5  # Promień wpływu

x_min = min(x)
x_max = max(x)
y_min = min(y)
y_max = max(y)

x_grid = np.linspace(x_min, x_max, 200)
y_grid = np.linspace(y_min, y_max, 200)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

# 6. Interpolacja mapy ciepła
points = np.column_stack((x, y))
values = np.ones_like(x)  # Jednolita intensywność
heatmap = griddata(points, values, (x_mesh, y_mesh), method='cubic', fill_value=0)

# 7. Wyświetlanie mapy ciepła
plt.figure(figsize=(10, 5))
plt.imshow(room_image, extent=[x_min, x_max, y_min, y_max])
plt.imshow(heatmap, extent=[x_min, x_max, y_min, y_max], origin='lower', alpha=0.5, cmap='hot')
plt.plot(x, y, 'ro')
plt.colorbar(label="Intensywność")
plt.title("Mapa ciepła odpowiedzi impulsowej (Ambisonics)")
plt.show()
