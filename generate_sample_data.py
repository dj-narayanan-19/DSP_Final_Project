from pydub import AudioSegment
from pydub.generators import WhiteNoise
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Folder with the MP3 files
mp3_folder = "../data/"
mp3_files = {
    "Puffin": "Puffinus_nativitatis.mp3",
    "Peregrin": "Falco_peregrinus.mp3",
    "Blue_Jay": "Cyanocitta_cristata.mp3",
}

# Output folder 
wav_folder = "./bird_sounds/"
os.makedirs(wav_folder, exist_ok=True)

# generate white noise
def generate_white_noise(duration, amplitude=-25):
    white_noise = WhiteNoise().to_audio_segment(duration=duration * 1000)
    return white_noise - abs(amplitude)

# bird sounds + white noise
def overlay_animal_sounds(background, animal_sounds, total_length):
    events = []
    full_audio = background
    for sound_file in animal_sounds:
        audio = AudioSegment.from_file(sound_file, format="wav")
        start_time = random.randint(0, total_length - int(len(audio) / 1000))
        full_audio = full_audio.overlay(audio, position=start_time * 1000)
        events.append({"timestamp": start_time, "file": sound_file})
    return full_audio, events

# plot spectrogram
def plot_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    n_fft = 1024
    hop_length = 512
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
    S_dB = librosa.amplitude_to_db(abs(S), ref=np.max)
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.show()


if __name__ == "__main__":
    np.random.seed(4810)
    # Ask the user for the total length of the track
    while True:
        try:
            total_length = int(input("Enter the total length of the audio track (1-100 seconds): "))
            if 1 <= total_length <= 100:
                break
            else:
                print("Please enter a number between 1 and 100.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    wav_files = []

    # Step 1: Convert MP3 files to WAV
    for bird, mp3_file in mp3_files.items():
        mp3_path = os.path.join(mp3_folder, mp3_file)
        wav_path = os.path.join(wav_folder, bird + ".wav")
        audio = AudioSegment.from_file(mp3_path, format="mp3")
        audio.export(wav_path, format="wav")
        print(f"Converted {mp3_path} to {wav_path}")
        wav_files.append(wav_path)

    # Step 2: Generate white noise
    background = generate_white_noise(total_length)

    # Step 3: Overlay bird sounds
    full_audio, events_log = overlay_animal_sounds(background, wav_files, total_length)

    # Step 4: Export final audio track
    output_file = "bird_audio_track.m4a"
    full_audio.export(output_file, format="wav")
    print(f"Audio track saved to {output_file}")

    # Step 5: Print event log
    print("Event Log:")
    for event in events_log:
        print(f"Bird Sound: {event['file']} | Timestamp: {event['timestamp']} sec")

    # Step 6: Plot spectrogram
    plot_spectrogram(output_file)