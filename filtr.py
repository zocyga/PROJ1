import numpy as np
from scipy.io.wavfile import read, write
import scipy.signal as signal

# Wczytaj plik WAV (odpowiedź impulsowa)
wav_path = "ambisonic_impulse_response.wav"
sample_rate, audio_data = read(wav_path)

if audio_data.ndim == 1:
    raise ValueError("Plik WAV ma tylko jeden kanał. Oczekiwano 4 kanałów B-formatu.")

# Oddziel składowe B-formatu
W = audio_data[:, 0]
X = audio_data[:, 1]
Y = audio_data[:, 2]
Z = audio_data[:, 3]
# Funkcja do wykrywania pierwszego odbicia
def separate_direct_and_reflection(signal, sample_rate, reflection_time_threshold=0.01):
    """
    Funkcja oddziela Direct Sound i First Reflection z odpowiedzi impulsowej.
    :param signal: Sygnal odpowiedzi impulsowej
    :param sample_rate: Częstotliwość próbkowania
    :param reflection_time_threshold: Minimalny czas między Direct Sound a First Reflection
    :return: Direct Sound oraz First Reflection
    """
    # Znajdź punkt początkowy sygnału (pierwszy impuls)
    direct_sound = signal[:int(sample_rate * 0.01)]  # Pierwsze 10 ms to Direct Sound

    # Druga część będzie traktowana jako First Reflection
    # Znajdź punkt, w którym zaczynają się odbicia (tutaj załóżmy, że refleksja pojawi się po około 10ms)
    start_reflection = int(sample_rate * 0.01) + int(sample_rate * reflection_time_threshold)

    # Podziel sygnał
    first_reflection = signal[start_reflection:]

    return direct_sound, first_reflection

# Rozdziel Direct Sound i First Reflection dla każdej składowej B-formatu
direct_W, reflection_W = separate_direct_and_reflection(W, sample_rate)
direct_X, reflection_X = separate_direct_and_reflection(X, sample_rate)
direct_Y, reflection_Y = separate_direct_and_reflection(Y, sample_rate)
direct_Z, reflection_Z = separate_direct_and_reflection(Z, sample_rate)

# Połącz składniki Direct Sound
direct_sound = np.column_stack([direct_W, direct_X, direct_Y, direct_Z])

# Połącz składniki First Reflection
first_reflection = np.column_stack([reflection_W, reflection_X, reflection_Y, reflection_Z])

# Zapisz pliki WAV
write("direct_sound.wav", sample_rate, (direct_sound * 32767).astype(np.int16))
write("first_reflection.wav", sample_rate, (first_reflection * 32767).astype(np.int16))

print("Pliki WAV zostały zapisane.")