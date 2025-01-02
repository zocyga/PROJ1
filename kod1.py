import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.interpolate import griddata

# 1. Wczytaj obraz panoramiczny
image_path = "obrazek.tiff"  # Zamień na swoją nazwę pliku
room_image = cv2.imread(image_path)
room_image = cv2.cvtColor(room_image, cv2.COLOR_BGR2RGB)

# 2. Wczytaj plik WAV (odpowiedź impulsowa)
wav_path = "impulse_response.wav"  # Zamień na swoją nazwę pliku
sample_rate, audio_data = read(wav_path)
audio_data = audio_data / np.max(np.abs(audio_data))  # Normalizacja

# 3. Ustaw mikrofon w centrum obrazu
height, width, _ = room_image.shape
microphone_position = (width // 2, height // 2)


# 4. Funkcja do obliczania azymutu i elewacji
def compute_azimuth_elevation(mic_position, reflection_position):
    dx = reflection_position[0] - mic_position[0]
    dy = reflection_position[1] - mic_position[1]

    # Azymut (kąt w poziomie)
    azimuth = np.arctan2(dy, dx)  # Kąt w radianach

    # Elewacja (kąt w pionie)
    distance = np.sqrt(dx ** 2 + dy ** 2)  # Odległość
    elevation = np.arctan2(dy, distance)  # Kąt w pionie

    # Konwersja na stopnie
    azimuth_deg = np.degrees(azimuth)  # Azymut w stopniach
    elevation_deg = np.degrees(elevation)  # Elewacja w stopniach

    return azimuth_deg, elevation_deg


# 5. Funkcja do symulacji odbić dźwięku
def simulate_sound_reflections(audio_data, mic_position, num_reflections=50):
    reflections = []
    for _ in range(num_reflections):
        # Losowa pozycja odbicia na ścianach pomieszczenia
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)

        # Odległość od mikrofonu
        distance = np.sqrt((x - mic_position[0]) ** 2 + (y - mic_position[1]) ** 2)

        # Amplituda dźwięku w tej pozycji (prosta zależność)
        amplitude = np.abs(audio_data[int(len(audio_data) * (distance / (width + height))) % len(audio_data)])

        # Obliczanie azymutu i elewacji
        azimuth, elevation = compute_azimuth_elevation(mic_position, (x, y))

        reflections.append((x, y, amplitude, azimuth, elevation))
    return reflections


# 6. Generowanie odbić
reflections = simulate_sound_reflections(audio_data, microphone_position)

# 7. Przygotowanie danych do interpolacji (mapa hipsometryczna)
x_points = np.array([r[0] for r in reflections])  # Pozycje X
y_points = np.array([r[1] for r in reflections])  # Pozycje Y
amplitudes = np.array([r[2] for r in reflections])  # Amplitudy

# Tworzenie siatki punktów dla mapy
grid_x, grid_y = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))

# Interpolacja amplitud na całej siatce
grid_amplitudes = griddata((x_points, y_points), amplitudes, (grid_x, grid_y), method='cubic', fill_value=0)

# 8. Wizualizacja
plt.figure(figsize=(12, 8))

# Podkład obrazu pomieszczenia
plt.imshow(room_image, extent=(0, width, height, 0))

# Mapa hipsometryczna amplitudy dźwięku
plt.imshow(grid_amplitudes, extent=(0, width, height, 0), cmap='hot', alpha=0.6)  # Kolory: odcienie czerwieni

# Mikrofon
plt.scatter(*microphone_position, color='blue', label='Mikrofon', s=200)

# Odbicia
for x, y, amplitude, azimuth, elevation in reflections:
    # Wizualizacja azymutu i elewacji
    plt.scatter(x, y, color='red', alpha=0.9, s=amplitude * 300)
    plt.text(x + 5, y + 5, f'Az: {azimuth:.1f}° El: {elevation:.1f}°', color='white', fontsize=8)

plt.legend()
plt.title("Wizualizacja odbić dźwięku w pomieszczeniu z mapą hipsometryczną")
plt.axis('off')
plt.colorbar(label='Amplituda dźwięku')
plt.show()
