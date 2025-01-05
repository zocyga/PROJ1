import numpy as np
from scipy.io.wavfile import write

# Parametry pliku WAV
sample_rate = 44100  # Częstotliwość próbkowania (Hz)
duration = 1.0       # Czas trwania (sekundy)
frequency = 440.0    # Częstotliwość sygnału (Hz)

# Generowanie sygnału sinusoidalnego dla każdej składowej ambisonicznej
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)  # Oś czasu

# Składowe B-formatu
S = 0.5*np.sin(2*np.pi*t*frequency)
W = S*(1/(np.sqrt(2)))
theta = np.pi/4 #horizontal angle
phi = np.pi/6 #elevation angle
#dźwiek z lewej strony i lekko z góry
X = S*np.cos(theta)*np.cos(phi)
Y = S*np.sin(theta)*np.cos(phi)
Z = S*np.sin(phi)

# Łączenie kanałów w jeden plik (format [N próbek x 4 kanały])
signal = np.stack([W, X, Y, Z], axis=1)

# Zapis do pliku WAV (16-bitowy format)
write("ambisonic_impulse_response.wav", sample_rate, (signal * 32767).astype(np.int16))

print("Plik ambisonic_impulse_response.wav został wygenerowany.")

