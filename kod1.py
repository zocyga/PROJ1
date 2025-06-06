import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.interpolate import griddata

# 1. Wczytaj obraz panoramiczny
image_path = "obrazek.tiff"  # Zamień na ścieżkę do obrazu panoramicznego
room_image = cv2.imread(image_path)  # Wczytanie obrazu
room_image = cv2.cvtColor(room_image, cv2.COLOR_BGR2RGB)  # Konwersja z BGR (OpenCV) na RGB (matplotlib)

# 2. Wczytaj plik WAV (odpowiedź impulsowa)
wav_path = "pliki_wave/impulse_response.wav"  # Ścieżka do pliku WAV
sample_rate, audio_data = read(wav_path)  # Odczyt pliku dźwiękowego
audio_data = audio_data / np.max(np.abs(audio_data))  # Normalizacja amplitudy dźwięku

# 3. Ustaw mikrofon w centrum obrazu
height, width, _ = room_image.shape  # Pobranie wymiarów obrazu
microphone_position = (width // 2, height // 2)  # Ustalenie pozycji mikrofonu w centrum obrazu

# 4. Funkcja do obliczania azymutu i elewacji
def compute_azimuth_elevation(mic_position, reflection_position):
    """
    Oblicza azymut i elewację odbicia względem pozycji mikrofonu.

    Args:
        mic_position (tuple): Pozycja mikrofonu w formacie (x, y).
        reflection_position (tuple): Pozycja odbicia w formacie (x, y).

    Returns:
        tuple: Azymut i elewacja w stopniach (azimuth_deg, elevation_deg).
    """
    dx = reflection_position[0] - mic_position[0]  # Różnica w osi X
    dy = reflection_position[1] - mic_position[1]  # Różnica w osi Y

    azimuth = np.arctan2(dy, dx)  # Obliczenie azymutu (kąt poziomy)
    distance = np.sqrt(dx ** 2 + dy ** 2)  # Odległość euklidesowa
    elevation = np.arctan2(dy, distance)  # Obliczenie elewacji (kąt pionowy)

    azimuth_deg = np.degrees(azimuth)  # Konwersja azymutu na stopnie
    elevation_deg = np.degrees(elevation)  # Konwersja elewacji na stopnie

    return azimuth_deg, elevation_deg  # Zwrócenie kątów w stopniach

# 5. Funkcja do symulacji odbić dźwięku
def simulate_sound_reflections(audio_data, mic_position, num_reflections=50):
    """
    Symuluje odbicia dźwięku w pomieszczeniu i oblicza ich parametry.

    Args:
        audio_data (np.array): Tablica danych audio (odpowiedź impulsowa).
        mic_position (tuple): Pozycja mikrofonu w formacie (x, y).
        num_reflections (int, optional): Liczba odbić do zasymulowania. Domyślnie 50.

    Returns:
        list: Lista krotek zawierających parametry odbić (x, y, amplitude, azimuth, elevation).
    """
    reflections = []  # Lista do przechowywania wyników odbić
    for _ in range(num_reflections):
        x = np.random.randint(0, width)  # Losowa pozycja X odbicia
        y = np.random.randint(0, height)  # Losowa pozycja Y odbicia

        distance = np.sqrt((x - mic_position[0]) ** 2 + (y - mic_position[1]) ** 2)  # Odległość od mikrofonu
        amplitude = np.abs(audio_data[int(len(audio_data) * (distance / (width + height))) % len(audio_data)])  # Skalowanie amplitudy

        azimuth, elevation = compute_azimuth_elevation(mic_position, (x, y))  # Obliczenie azymutu i elewacji
        reflections.append((x, y, amplitude, azimuth, elevation))  # Dodanie odbicia do listy
    return reflections  # Zwrócenie listy odbić

# 6. Generowanie odbić
reflections = simulate_sound_reflections(audio_data, microphone_position)  # Symulacja odbić

# 7. Przygotowanie danych do interpolacji (mapa hipsometryczna)
x_points = np.array([r[0] for r in reflections])  # Pozycje X odbić
y_points = np.array([r[1] for r in reflections])  # Pozycje Y odbić
amplitudes = np.array([r[2] for r in reflections])  # Amplitudy odbić

# Tworzenie siatki punktów dla mapy
grid_x, grid_y = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))  # Generowanie siatki

# Interpolacja amplitud na całej siatce
grid_amplitudes = griddata((x_points, y_points), amplitudes, (grid_x, grid_y), method='cubic', fill_value=0)  # Interpolacja

# 8. Wizualizacja
plt.figure(figsize=(12, 8))  # Rozmiar wykresu
plt.imshow(room_image, extent=(0, width, height, 0))  # Wyświetlenie obrazu pomieszczenia jako tło

plt.imshow(grid_amplitudes, extent=(0, width, height, 0), cmap='hot', alpha=0.6)  # Nałożenie mapy hipsometrycznej amplitudy dźwięku
plt.scatter(*microphone_position, color='blue', label='Mikrofon', s=200)  # Pozycja mikrofonu

# Odbicia dźwięku (wizualizacja)
for x, y, amplitude, azimuth, elevation in reflections:
    plt.scatter(x, y, color='red', alpha=0.9, s=amplitude * 300)  # Punkty odbić
    plt.text(x + 5, y + 5, f'Az: {azimuth:.1f}° El: {elevation:.1f}°', color='white', fontsize=8)  # Opisy azymutu i elewacji

plt.legend()  # Legenda
plt.title("Wizualizacja odbić dźwięku w pomieszczeniu z mapą hipsometryczną")  # Tytuł wykresu
plt.axis('off')  # Usunięcie osi
plt.colorbar(label='Amplituda dźwięku')  # Pasek kolorów
plt.show()  # Wyświetlenie wykresu