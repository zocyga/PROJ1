import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
import librosa


# Funkcja do normalizacji sygnału
def normalize(audio):
    return audio / np.max(np.abs(audio))


# Wczytanie pliku audio i dostosowanie częstotliwości próbkowania
# Częstotliwość próbkowania jest dostosowana dla pliku RIR, ponieważ z założenia jest krótszy
def load_audio_resample(file_path, target_samplerate):
    data, samplerate = sf.read(file_path)
    if samplerate != target_samplerate:
        print(f"Resampling {file_path} from {samplerate} Hz to {target_samplerate} Hz...")
        data = librosa.resample(data, orig_sr=samplerate, target_sr=target_samplerate)
        samplerate = target_samplerate
    return data, samplerate


# Zapisz przetworzony plik audio
def save_audio(file_path, data, samplerate):
    sf.write(file_path, data, samplerate)


# Funkcja do realizacji efektu convolutional reverb w stereo
def apply_convolutional_reverb_stereo(audio_file, rir_file, output_file):
    # Wczytaj utwór
    audio, samplerate_audio = sf.read(audio_file)

    # Wczytaj odpowiedź impulsową (IR)
    rir, samplerate_rir = sf.read(rir_file)

    # Jeśli częstotliwości próbkowania IR różnią się od utworu, dostosuj je
    if samplerate_rir != samplerate_audio:
        rir, _ = load_audio_resample(rir_file, samplerate_audio)

    # Jeśli utwór jest mono, powiel kanały do stereo
    if len(audio.shape) == 1:
        audio = np.column_stack((audio, audio))

    # Jeśli RIR jest mono, powiel kanały do stereo
    if len(rir.shape) == 1:
        rir = np.column_stack((rir, rir))

    # Splot dla każdego kanału stereo
    convolved_left = fftconvolve(audio[:, 0], rir[:, 0], mode='full')
    convolved_right = fftconvolve(audio[:, 1], rir[:, 1], mode='full')

    # Połączenie przetworzonych kanałów w jeden sygnał stereo
    convolved_stereo = np.column_stack((convolved_left, convolved_right))

    # Normalizacja wynikowego sygnału
    convolved_stereo = normalize(convolved_stereo)

    # Zapisz wynik do pliku
    save_audio(output_file, convolved_stereo, samplerate_audio)
    print(f"Przetworzony plik audio zapisano jako {output_file}")


# Realizacja efektu reverb
if __name__ == "__main__":
    # Ścieżki do plików audio
    audio_file = "inputaudio.wav"  # Plik utworu
    rir_file = "AirRaidShelter.wav"  # Plik odpowiedzi impulsowej
    output_file = "output_shelter.wav"  # Wyjściowy plik z efektem

    # Nałóż efekt convolutional reverb w stereo
    apply_convolutional_reverb_stereo(audio_file, rir_file, output_file)
