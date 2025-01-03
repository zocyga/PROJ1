import numpy as np
from scipy.io.wavfile import write

# Parametry pliku WAV
sample_rate = 44100  # Częstotliwość próbkowania (Hz)
duration = 1.0       # Czas trwania (sekundy)
frequency = 440.0    # Częstotliwość sygnału (Hz)

# Generowanie sygnału sinusoidalnego dla każdej składowej ambisonicznej
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)  # Oś czasu

# Składowe B-formatu
W = 0.5 * np.sin(2 * np.pi * frequency * t)  # Wszechkierunkowy
X = 0.5 * np.sin(2 * np.pi * frequency * t + np.pi / 4)  # Przesunięcie fazowe 45 stopni
Y = 0.5 * np.sin(2 * np.pi * frequency * t + np.pi / 2)  # Przesunięcie o 90 stopni
Z = 0.5 * np.sin(2 * np.pi * frequency * t + np.pi / 3)  # Przesunięcie o 60 stopni

# Łączenie kanałów w jeden plik (format [N próbek x 4 kanały])
signal = np.stack([W, X, Y, Z], axis=1)

# Zapis do pliku WAV (16-bitowy format)
write("ambisonic_impulse_response.wav", sample_rate, (signal * 32).astype(np.int16))

print("Plik ambisonic_impulse_response.wav został wygenerowany.")

