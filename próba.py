import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import griddata

# 1. Wczytanie odpowiedzi impulsowej
filename = 'M0001_S01_R04_ambix.wav'
y, sr = librosa.load(filename, sr=None, mono=False)  # Załaduj wszystkie kanały

# Sprawdzenie liczby kanałów
y_w, y_x, y_y, y_z = y  # Kanały ambisoniczne (W, X, Y, Z)

# 2. Wczytanie sferycznego obrazu
image_filename = 'S1_R4_sferyczne.jpg'
room_image = cv2.imread(image_filename)
room_image = cv2.cvtColor(room_image, cv2.COLOR_BGR2RGB)  # Konwersja do RGB

height, width, _ = room_image.shape  # Wymiary obrazu
microphone_position = (3270, 1242)  # Współrzędne mikrofonu na obrazie

# Parametry dźwięku
c = 343  # Prędkość dźwięku w powietrzu (m/s)

# 3. Podział odpowiedzi impulsowej na ramki 15 ms
frame_size = int(0.015 * sr)  # 15 ms w próbkach
num_frames = len(y_w) // frame_size


# 4. Obliczanie azymutu i elewacji
def compute_azimuth_elevation(mic_position, reflection_position):
    dx = reflection_position[0] - mic_position[0]
    dy = reflection_position[1] - mic_position[1]
    azimuth = np.arctan2(dy, dx)
    distance = np.sqrt(dx ** 2 + dy ** 2)
    elevation = np.arctan2(dy, distance)
    azimuth_deg = np.degrees(azimuth)
    elevation_deg = np.degrees(elevation)
    return azimuth_deg, elevation_deg


# 5. Symulacja odbić dźwięku
def simulate_sound_reflections(audio_data, mic_position, num_reflections=50):
    reflections = []
    for _ in range(num_reflections):
        x = np.random.randint(0, width)  # Losowa pozycja X odbicia
        y = np.random.randint(0, height)  # Losowa pozycja Y odbicia
        distance = np.sqrt((x - mic_position[0]) ** 2 + (y - mic_position[1]) ** 2)  # Odległość od mikrofonu
        amplitude = np.abs(
            audio_data[int(len(audio_data) * (distance / (width + height))) % len(audio_data)])  # Skalowanie amplitudy
        azimuth, elevation = compute_azimuth_elevation(mic_position, (x, y))  # Obliczenie azymutu i elewacji
        reflections.append((x, y, amplitude, azimuth, elevation))  # Dodanie odbicia do listy
    return reflections


# Generowanie odbić
reflections = simulate_sound_reflections(y_w, microphone_position)  # Symulacja odbić

# 6. Przygotowanie danych do interpolacji (mapa hipsometryczna)
x_points = np.array([r[0] for r in reflections])  # Pozycje X odbić
y_points = np.array([r[1] for r in reflections])  # Pozycje Y odbić
amplitudes = np.array([r[2] for r in reflections])  # Amplitudy odbić

# Tworzenie siatki punktów dla mapy
grid_x, grid_y = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))  # Generowanie siatki

# Interpolacja amplitud na całej siatce
grid_amplitudes = griddata((x_points, y_points), amplitudes, (grid_x, grid_y), method='cubic', fill_value=0)

# 7. Przygotowanie wykresu
fig, ax = plt.subplots(figsize=(12, 8))  # Rozmiar wykresu
ax.imshow(room_image, extent=(0, width, height, 0))  # Wyświetlenie obrazu pomieszczenia jako tło

# Nałożenie mapy amplitudy dźwięku z przezroczystością
heatmap = ax.imshow(grid_amplitudes, extent=(0, width, height, 0), cmap='hot', alpha=0.4)  # Zmniejszona przezroczystość

# Odbicia dźwięku (wizualizacja)
scatter = ax.scatter([], [], color='red', alpha=0.9, s=0)  # Puste początkowe odbicia

# Podział na ramki czasowe
frame_times = np.arange(0, len(y_w) / sr, 0.015)  # Czas dla każdej ramki 15 ms
current_frame = 0  # Indeks bieżącej ramki


# Funkcja aktualizująca wizualizację
def update_frame(frame):
    global current_frame
    current_frame = frame

    start_idx = current_frame * frame_size
    end_idx = (current_frame + 1) * frame_size
    energy = np.sum(y_w[start_idx:end_idx] ** 2)  # Obliczanie energii w ramce

    # Przygotowanie nowych odbić
    x_vals = []
    y_vals = []
    sizes = []
    for x, y, amplitude, _, _ in reflections:
        x_vals.append(x)
        y_vals.append(y)
        sizes.append(amplitude * 1000)  # Skala wielkości punktów na mapie

    # Aktualizacja scatter plot z odbiciami
    scatter.set_offsets(np.c_[x_vals, y_vals])
    scatter.set_sizes(sizes)

    # Zaktualizowanie tytułu
    ax.set_title(f"Ramka czasowa: {current_frame + 1}/{num_frames}")

    # Rysowanie podziału na ramki czasowe
    for t in frame_times:
        ax.axvline(x=t * width / (len(y_w) / sr), color='blue', linestyle='--', alpha=0.5)  # Linia dla każdej ramki

    plt.draw()


# Funkcja obsługująca klawisze
def on_key(event):
    global current_frame

    if event.key == 'right' and current_frame < num_frames - 1:
        current_frame += 1
        update_frame(current_frame)

    elif event.key == 'left' and current_frame > 0:
        current_frame -= 1
        update_frame(current_frame)


# Ustawienie interaktywności
fig.canvas.mpl_connect('key_press_event', on_key)

# Pierwsza aktualizacja
update_frame(current_frame)

# Wyświetlenie wizualizacji
plt.show()
