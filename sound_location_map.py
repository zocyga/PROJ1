import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
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

# # 3. Oblicz amplitudę (maksymalna wartość W)
amplitudes= []
for _ in W:
    _ = np.abs(np.int64(_))
    amplitudes.append (_)

amplitude = np.max(W)
height, width, _ = room_image.shape  # Pobranie wymiarów obrazu
mic_position = (width // 2, height // 2)  # Ustalenie pozycji mikrofonu w centrum obrazu
# 4. Oblicz kierunek (azymut i elewacja)
Ix = np.int64(W * X)
Iy = np.int64(W * Y)
Iz = np.int64(W * Z)

def compute_azimuth_elevation(Ix, Iy, Iz):
    azimuth = np.arctan2(Iy, Ix)
    distance = np.sqrt(((Ix - mic_position[0]) ** 2) + ((Iy - mic_position[1]) ** 2))
    elevation = np.arctan2(Iz, distance)
    azimuth_deg = np.degrees(azimuth)
    elevation_deg = np.degrees(elevation)

    return azimuth_deg, elevation_deg, distance


azimuth, elevation, distance = compute_azimuth_elevation(Ix, Iy, Iz)


def sound_reflections(X, Y, amplitudes, azimuth, elevation):
    reflections = []  # Lista do przechowywania wyników odbić
    for i in range(len(X)):
        reflections.append((X[i], Y[i], amplitudes[i], azimuth[i], elevation[i]))  # Dodanie odbicia do listy
    return reflections  # Zwrócenie listy odbić

reflections = sound_reflections(X,Y, amplitudes, azimuth, elevation)
x_points = np.array([r[0] for r in reflections])  # Pozycje X odbić
y_points = np.array([r[1] for r in reflections])  # Pozycje Y odbić
amplitudes1 = np.array([r[2] for r in reflections])  # Amplitudy odbić



grid_x, grid_y = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))  # Generowanie siatki

#Interpolacja amplitud na całej siatce
grid_amplitudes = griddata((x_points, y_points), amplitudes1, (grid_x, grid_y), method='cubic', fill_value=0)  # Interpolacja
"""
# 8. Wizualizacja
plt.figure(figsize=(12, 8))  # Rozmiar wykresu
plt.imshow(room_image, extent=(0, width, height, 0))  # Wyświetlenie obrazu pomieszczenia jako tło

plt.imshow(grid_amplitudes, extent=(0, width, height, 0), cmap='hot', alpha=0.6)  # Nałożenie mapy hipsometrycznej amplitudy dźwięku
plt.scatter(*mic_position, color='blue', label='Mikrofon', s=200)  # Pozycja mikrofonu

# Odbicia dźwięku (wizualizacja)
for x, y, amplitude, azimuth, elevation in reflections:
    plt.scatter(x, y, color='red', alpha=0.9, s=amplitude * 300)  # Punkty odbić
    plt.text(x + 5, y + 5, f'Az: {azimuth:.1f}° El: {elevation:.1f}°', color='white', fontsize=8)  # Opisy azymutu i elewacji

plt.legend()  # Legenda
plt.title("Wizualizacja odbić dźwięku w pomieszczeniu z mapą hipsometryczną")  # Tytuł wykresu
plt.axis('off')  # Usunięcie osi
plt.colorbar(label='Amplituda dźwięku')  # Pasek kolorów
plt.show()  # Wyświetlenie wykresu
"""
#print(W, X,Y,Z)