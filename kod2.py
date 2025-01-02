import numpy as np
from scipy.io.wavfile import write

# Parametry pliku WAV
sample_rate = 44100  # Częstotliwość próbkowania (Hz)
duration = 1.0       # Czas trwania (sekundy)
frequency = 440.0    # Częstotliwość sygnału (Hz, np. dźwięk A4)

# Generowanie sygnału sinusoidalnego
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)  # Oś czasu
signal = 0.5 * np.sin(2 * np.pi * frequency * t)  # Sygnał sinusoidalny (amplituda 0.5)

# Zapis do pliku WAV
write("impulse_response.wav", sample_rate, (signal * 32767).astype(np.int16))  # Normalizacja do 16-bitowego formatu
print("Plik impulse_response.wav został wygenerowany.")
