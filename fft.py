import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Wczytanie pliku WAV
file_path = "pliki_wave/direct_sound.wav"  # Podaj ścieżkę do pliku WAV
sample_rate, data = wavfile.read(file_path)

# Sprawdzenie liczby kanałów
if len(data.shape) == 1:  # Mono
    print("Plik jest mono. Przetwarzam pojedynczy kanał.")
    channel_data = {"channel_1": data}  # Jedyny kanał w pliku mono
elif data.shape[1] < 4:  # Plik ma mniej niż 4 kanały
    print(f"Plik zawiera tylko {data.shape[1]} kanały, oczekiwano 4.")
    exit()
else:  # Więcej niż 1 kanał
    print("Plik jest wielokanałowy. Przetwarzam 4 kanały.")
    channel_data = {f"channel_{i+1}": data[:, i] for i in range(4)}

# Obliczenie FFT dla każdego kanału
fft_results = {}
frequencies = np.fft.rfftfreq(data.shape[0], d=1/sample_rate)  # Oś częstotliwości

# Oblicz FFT dla każdego kanału
for channel, response in channel_data.items():
    fft_results[channel] = np.fft.rfft(response)  # FFT dla danego kanału

# Compute the combined amplitude (sum of the magnitudes of all channels)
combined_amplitude = np.zeros_like(frequencies)
for fft_data in fft_results.values():
    combined_amplitude += np.abs(fft_data)

# Wizualizacja wyników
plt.figure(figsize=(12, 8))

# Tworzenie wykresu dla każdego kanału
for channel, fft_data in fft_results.items():
    plt.plot(frequencies, 20 * np.log10(np.abs(fft_data)), label=channel)

# Plot the combined amplitude (sum of all channels)
plt.plot(frequencies, 20 * np.log10(combined_amplitude), label="Combined Amplitude", linestyle='--', color='black')

# Konfiguracja wykresu
plt.title("FFT dla odpowiedzi impulsowej (Mono lub 4 kanały) z sumą amplitud")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda (dB)")
plt.legend()  # Legenda, by odróżnić kanały
plt.grid(True)  # Dodanie siatki dla lepszej czytelności
plt.tight_layout()  # Dopasowanie rozmiaru wykresu do zawartości

# Wyświetlenie wykresu
plt.show()
