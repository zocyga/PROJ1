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


def generate_wav(filename, sample_rate=44100, duration=1.0, frequency=440.0):
    """
    Generuje plik WAV z sygnałem sinusoidalnym o zadanej częstotliwości.

    Args:
        filename (str): Nazwa pliku wynikowego.
        sample_rate (int, optional): Częstotliwość próbkowania w Hz. Domyślnie 44100 Hz.
        duration (float, optional): Czas trwania sygnału w sekundach. Domyślnie 1 sekunda.
        frequency (float, optional): Częstotliwość sygnału sinusoidalnego w Hz. Domyślnie 440 Hz.

    Returns:
        None
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)  # Oś czasu
    signal = 0.5 * np.sin(2 * np.pi * frequency * t)  # Sygnał sinusoidalny (amplituda 0.5)
    write(filename, sample_rate, (signal * 32767).astype(np.int16))  # Normalizacja do 16-bitowego formatu
    print(f"Plik {filename} został wygenerowany.")

# Przykładowe wywołanie funkcji
generate_wav("impulse_response.wav")
